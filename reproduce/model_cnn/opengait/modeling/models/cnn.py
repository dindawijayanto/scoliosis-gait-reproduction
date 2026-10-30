import numpy as np
import torch
import torch.nn as nn

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, PackSequenceWrapper

from einops import rearrange

LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}


class CNN_MT(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.TP = PackSequenceWrapper(torch.max)
        self.FC = nn.Linear(model_cfg['in_channels'], model_cfg['embed_dim'], bias=False)
        self.Classifier = nn.Linear(model_cfg['embed_dim'], model_cfg['class_num'], bias=False)

    def forward(self, inputs):
        ipts, pids, labels, _, seqL = inputs

        label_ids = np.array([LABEL_MAP[status] for status in labels])
        label_ids = torch.from_numpy(label_ids).cuda().long()

        sils = ipts[0]
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        del ipts
        outs = self.Backbone(sils)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w]

        # Global pooling over the whole map
        n, c = outs.size()[:2]
        z = outs.view(n, c, -1)
        feat = z.mean(-1) + z.max(-1)[0]  # [n, c]

        embed = self.FC(feat)  # [n, 256]
        logits = self.Classifier(embed)  # [n, 3]

        embed = embed.unsqueeze(-1)  # [n, 256, 1]
        logits = logits.unsqueeze(-1)  # [n, 3, 1]
        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed, 'labels': self.triplet_labels(pids, label_ids)},
                'softmax': {'logits': logits, 'labels': label_ids},
            },
            'visual_summary': {
                'image/sils': rearrange(sils, 'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': logits
            }
        }
        return retval

    def triplet_labels(self, pids, label_ids):
        return pids


class CNN_Triplet(CNN_MT):

    def triplet_labels(self, pids, label_ids):
        return label_ids


class CNN_CE(CNN_MT):

    def forward(self, inputs):
        retval = super().forward(inputs)
        del retval['training_feat']['triplet']
        return retval
