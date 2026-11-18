import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

CLASS_ORDER = ['negative', 'neutral', 'positive']


class ScoNetRobust(BaseModel):
    """ScoNet-MT (ResNet9 + max over frames + 16 strips + triplet & CE) with optional parts for exp_2.

    model_cfg keys on top of ScoNet's:
      aux_head: none | bam | ordinal
        bam     : one binary head "neutral vs rest" (TG-MILNet's BAM, here with the LA loss).
        ordinal : two binary heads "y >= neutral" and "y >= positive" (negative < neutral < positive).
                  The neutral/negative border then gets gradient from every scoliosis child.
      Every head is a SeparateBNNecks on the same 16-strip embedding as the main head.

    Losses read their class prior from the training sampler (ClassAwareSampler.class_prior, or the
    natural class frequencies for TripletSampler), so the LA loss knows what the batches look like.
    inference_feat: 'embeddings' = main logits [n, 3, p] (OpenGait's evaluator uses only these),
                    'aux' = auxiliary logits [n, tasks, 2, p] (exported for val-chosen fusion).
    """

    def build_network(self, model_cfg):
        self.Backbone = SetBlockWrapper(self.get_backbone(model_cfg['backbone_cfg']))
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])

        self.aux_type = model_cfg.get('aux_head', 'none')
        n_tasks = {'none': 0, 'bam': 1, 'ordinal': 2}[self.aux_type]
        neck = dict(model_cfg['SeparateBNNecks'], class_num=2)
        self.AuxNecks = nn.ModuleList([SeparateBNNecks(**neck) for _ in range(n_tasks)])
        self._log_priors = None

    # ---- class priors seen by the losses -------------------------------------------------
    def _class_prior(self):
        sampler = self.train_loader.batch_sampler
        if hasattr(sampler, 'class_prior'):
            return list(sampler.class_prior)
        ds = self.train_loader.dataset             # TripletSampler: uniform over children
        counts = [0, 0, 0]
        for pid in ds.label_set:
            counts[CLASS_ORDER.index(ds.types_list[ds.indices_dict[pid][0]].lower())] += 1
        return [c / sum(counts) for c in counts]

    def log_priors(self):
        if self._log_priors is None:
            p = self._class_prior()
            binary = {'bam': [1 - p[1], p[1]], 'ord1': [p[0], p[1] + p[2]], 'ord2': [p[0] + p[1], p[2]]}
            self._log_priors = {k: torch.tensor([math.log(max(x, 1e-8)) for x in v]).cuda()
                                for k, v in dict(main=p, **binary).items()}
            self.msg_mgr.log_info('class prior seen by the losses (neg, neu, pos): %s' % [round(x, 4) for x in p])
        return self._log_priors

    # ---- forward ---------------------------------------------------------------------------
    def forward(self, inputs):
        ipts, pids, labels, _, seqL = inputs
        label_ids = torch.from_numpy(np.array([CLASS_ORDER.index(s.lower()) for s in labels])).cuda().long()

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')
        del ipts

        outs = self.Backbone(sils)                                # [n, c, s, h, w]
        outs = self.TP(outs, seqL, options={"dim": 2})[0]         # [n, c, h, w]
        feat = self.HPP(outs)                                     # [n, c, p]
        embed = self.FCs(feat)                                    # [n, c, p]
        _, logits = self.BNNecks(embed)                           # [n, 3, p]
        aux = [neck(embed)[1] for neck in self.AuxNecks]          # each [n, 2, p]

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': pids},
            },
            'visual_summary': {'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')},
            'inference_feat': {'embeddings': logits},
        }
        if aux:
            retval['inference_feat']['aux'] = torch.stack(aux, 1)  # [n, tasks, 2, p]

        if self.training:
            lp = self.log_priors()
            tf = retval['training_feat']
            tf['softmax'] = {'logits': logits, 'labels': label_ids, 'log_prior': lp['main']}
            if self.aux_type == 'bam':
                tf['bam'] = {'logits': aux[0], 'labels': (label_ids == 1).long(), 'log_prior': lp['bam']}
            elif self.aux_type == 'ordinal':
                tf['ord1'] = {'logits': aux[0], 'labels': (label_ids >= 1).long(), 'log_prior': lp['ord1']}
                tf['ord2'] = {'logits': aux[1], 'labels': (label_ids >= 2).long(), 'log_prior': lp['ord2']}
        return retval
