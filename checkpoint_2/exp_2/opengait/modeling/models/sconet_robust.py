import math

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, SeparateFCs, SeparateBNNecks

CLASS_ORDER = ['negative', 'neutral', 'positive']


class GatedAttention(nn.Module):
    """Per-strip gated attention over frames (checkpoint_2 notebook 04). Starts as a plain mean over frames."""

    def __init__(self, c=512, hidden=128):
        super().__init__()
        self.V = nn.Linear(c, hidden)
        self.U = nn.Linear(c, hidden)
        self.w = nn.Linear(hidden, 1)

    def zero_init(self):
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, h):                                   # h [n, s, c, p]
        x = h.permute(0, 1, 3, 2).float()                   # [n, s, p, c]
        e = self.w(torch.tanh(self.V(x)) * torch.sigmoid(self.U(x))).squeeze(-1)   # [n, s, p]
        alpha = torch.softmax(e, dim=1)
        return (alpha.unsqueeze(2) * h.float()).sum(1), alpha                       # [n, c, p]


class DynamicStripWeights(nn.Module):
    """Per-child, per-class weights over the 16 strips (checkpoint_2 notebook 03). Starts as the ScoNet mean."""

    def __init__(self, embed_dim=256, classes=3, hidden=64):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(embed_dim, hidden), nn.Tanh(), nn.Linear(hidden, classes))

    def zero_init(self):
        nn.init.zeros_(self.score[2].weight)
        nn.init.zeros_(self.score[2].bias)

    def forward(self, embed, logits):                       # embed [n, c, p], logits [n, 3, p]
        a = self.score(embed.permute(0, 2, 1).float())      # [n, p, 3]
        w = torch.softmax(a.permute(0, 2, 1), dim=-1)       # [n, 3, p], softmax over strips
        return (logits.float() * w).sum(-1, keepdim=True)   # [n, 3, 1]


class ScoNetRobust(BaseModel):
    """ScoNet-MT (ResNet9 + max over frames + 16 strips + triplet & CE) with optional parts for exp_2.
    Every option off = exactly ScoNet-MT.

    model_cfg keys on top of ScoNet's:
      aux_head: none | bam | ordinal
        bam     : binary head "neutral vs rest" (TG-MILNet's BAM, here with the LA loss)
        ordinal : two binary heads "y >= neutral" and "y >= positive" (negative < neutral < positive)
      temporal: max | attention       frame pooling; attention = gated attention per strip (notebook 04)
      strip_weights: none | dynamic   learned per-child strip weights for the decision (notebook 03)
      gei_channel: false | true       2nd input channel = frame - mean of its sequence (ScoliGaitX GEI-subtract,
                                      early fusion as in notebook 05; needs backbone_cfg.in_channel = 2)
      train_flip: false | true        random horizontal flip of whole training sequences (notebook 07)
      class_triplet: false | true     extra triplet loss with the CLASS as identity (notebook 07)
    Losses read their class prior from the training sampler, so the LA loss knows what the batches look like.
    inference_feat: 'embeddings' = decision logits [n, 3, p] (OpenGait's evaluator and calibrate.py average
                    over p; with strip weights p = 1 and the weighted logits are used),
                    'aux' = auxiliary logits [n, tasks, 2, p] for val-chosen fusion.
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

        self.temporal = model_cfg.get('temporal', 'max')
        assert self.temporal in ('max', 'attention')
        if self.temporal == 'attention':
            self.FrameAttention = GatedAttention(c=model_cfg['SeparateFCs']['in_channels'])
        self.strip_mode = model_cfg.get('strip_weights', 'none')
        assert self.strip_mode in ('none', 'dynamic')
        if self.strip_mode == 'dynamic':
            self.StripWeights = DynamicStripWeights(embed_dim=model_cfg['SeparateFCs']['out_channels'])
        self.gei_channel = model_cfg.get('gei_channel', False)
        self.train_flip = model_cfg.get('train_flip', False)
        self.class_triplet = model_cfg.get('class_triplet', False)
        self._log_priors = None

    def init_parameters(self):
        super().init_parameters()
        for name in ('FrameAttention', 'StripWeights'):        # new modules start as plain ScoNet
            if hasattr(self, name):
                getattr(self, name).zero_init()

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
    def _pool_frames(self, outs, seqL):
        """outs [n, c, s, h, w] -> strip features [n, c, p]"""
        if self.temporal == 'max':
            return self.HPP(self.TP(outs, seqL, options={"dim": 2})[0])
        if seqL is not None and len(seqL[0]) > 1:
            raise ValueError('frame attention needs one sequence per batch at test time (evaluator batch_size 1)')
        n, c, s = outs.shape[:3]
        h = self.HPP(rearrange(outs, 'n c s h w -> (n s) c h w'))          # [(n s), c, p]
        h = rearrange(h, '(n s) c p -> n s c p', n=n, s=s)
        pooled, _ = self.FrameAttention(h)
        return pooled.to(outs.dtype)

    def forward(self, inputs):
        ipts, pids, labels, _, seqL = inputs
        label_ids = torch.from_numpy(np.array([CLASS_ORDER.index(s.lower()) for s in labels])).cuda().long()

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')
        del ipts
        if self.training and self.train_flip:                               # whole sequence, 50%
            mirror = torch.rand(sils.shape[0], device=sils.device) < 0.5
            sils = torch.where(mirror.view(-1, 1, 1, 1, 1), sils.flip(-1), sils)
        if self.gei_channel:                                                # frame - mean of its own sequence
            sils = torch.cat([sils, sils - sils.mean(2, keepdim=True)], 1)  # [n, 2, s, h, w]

        outs = self.Backbone(sils)                                # [n, c, s, h, w]
        feat = self._pool_frames(outs, seqL)                      # [n, c, p]
        embed = self.FCs(feat)                                    # [n, c, p]
        _, logits = self.BNNecks(embed)                           # [n, 3, p]
        aux = [neck(embed)[1] for neck in self.AuxNecks]          # each [n, 2, p]
        decision = self.StripWeights(embed, logits) if self.strip_mode == 'dynamic' else logits

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': pids},
            },
            'visual_summary': {'image/sils': rearrange(sils[:, :1], 'n c s h w -> (n s) c h w')},
            'inference_feat': {'embeddings': decision},
        }
        if aux:
            retval['inference_feat']['aux'] = torch.stack(aux, 1)  # [n, tasks, 2, p]

        if self.training:
            lp = self.log_priors()
            tf = retval['training_feat']
            tf['softmax'] = {'logits': logits, 'labels': label_ids, 'log_prior': lp['main']}
            if self.strip_mode == 'dynamic':
                tf['strip'] = {'logits': decision, 'labels': label_ids, 'log_prior': lp['main']}
            if self.class_triplet:
                tf['class_triplet'] = {'embeddings': embed, 'labels': label_ids}
            if self.aux_type == 'bam':
                tf['bam'] = {'logits': aux[0], 'labels': (label_ids == 1).long(), 'log_prior': lp['bam']}
            elif self.aux_type == 'ordinal':
                tf['ord1'] = {'logits': aux[0], 'labels': (label_ids >= 1).long(), 'log_prior': lp['ord1']}
                tf['ord2'] = {'logits': aux[1], 'labels': (label_ids >= 2).long(), 'log_prior': lp['ord2']}
        return retval
