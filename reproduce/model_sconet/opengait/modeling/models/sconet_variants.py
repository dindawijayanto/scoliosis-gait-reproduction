import numpy as np
import torch

from .sconet import ScoNet

LABEL_MAP = {'negative': 0, 'neutral': 1, 'positive': 2}


class ScoNet_CE(ScoNet):

    def forward(self, inputs):
        retval = super().forward(inputs)
        # LossAggregator raises if a training_feat key has no loss in loss_cfg,
        # so the triplet entry is removed instead of given weight 0.
        del retval['training_feat']['triplet']
        return retval


class ScoNet_Triplet(ScoNet):

    def forward(self, inputs):
        retval = super().forward(inputs)
        labels = inputs[2]
        label_ids = np.array([LABEL_MAP[status] for status in labels])
        label_ids = torch.from_numpy(label_ids).cuda().long()
        retval['training_feat']['triplet']['labels'] = label_ids
        return retval
