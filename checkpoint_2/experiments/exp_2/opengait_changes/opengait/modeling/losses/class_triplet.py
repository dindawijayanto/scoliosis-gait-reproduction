import torch
import torch.nn.functional as F

from .base import BaseLoss


class ClassTripletLoss(BaseLoss):
    """Triplet loss with the CLASS as identity (checkpoint_2 notebook 07), per strip.

    anchor + positive = same class (not the same item), negative = other class; batch-all, averaged over the
    non-zero triplets. Unlike OpenGait's TripletLoss it does not need P x K batches, so the classes can have any
    count. A batch with one class only gives exactly 0.
    """

    def __init__(self, margin=0.2, loss_term_weight=1.0):
        super().__init__(loss_term_weight)
        self.margin = margin

    def forward(self, embeddings, labels):
        x = embeddings.permute(2, 0, 1).float()                         # [p, n, c]
        x2 = (x ** 2).sum(-1)
        dist = torch.sqrt(F.relu(x2.unsqueeze(2) + x2.unsqueeze(1) - 2 * x.matmul(x.transpose(1, 2))) + 1e-12)
        n = labels.shape[0]
        same = labels.unsqueeze(1) == labels.unsqueeze(0)
        eye = torch.eye(n, dtype=torch.bool, device=labels.device)
        valid = (same & ~eye).unsqueeze(2) & (~same).unsqueeze(1)       # [n, n, n] anchor, positive, negative
        loss = F.relu(dist.unsqueeze(3) - dist.unsqueeze(2) + self.margin) * valid.unsqueeze(0)
        loss = loss.view(loss.shape[0], -1)
        num = (loss > 0).sum(-1).float()
        avg = (loss.sum(-1) / (num + 1e-9)).mean()
        self.info.update({'loss': avg.detach().clone(), 'loss_num': num.mean().detach().clone()})
        return avg, self.info
