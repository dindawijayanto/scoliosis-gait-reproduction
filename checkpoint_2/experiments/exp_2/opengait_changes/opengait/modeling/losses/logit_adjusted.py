import torch
import torch.nn.functional as F

from .base import BaseLoss


class LogitAdjustedCE(BaseLoss):
    """Logit-adjusted cross entropy (Menon et al., ICLR 2021) on part-wise logits.

    loss = CE(scale * logits + tau * log_prior, labels), applied to every strip.
    log_prior is the class distribution the loss sees during training (the sampler's prior), so
    with a balanced sampler the offset is a constant and this is plain CE. At test time the raw
    logits are used, which targets balanced accuracy. tau=0 gives ScoNet's CrossEntropyLoss.
    Works for any number of classes, so it also trains the binary auxiliary heads.
    """

    def __init__(self, scale=16, tau=1.0, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super().__init__(loss_term_weight)
        self.scale, self.tau = scale, tau
        self.eps = eps if label_smooth else 0.0
        self.log_accuracy = log_accuracy

    def forward(self, logits, labels, log_prior):
        """logits [n, c, p], labels [n], log_prior [c]"""
        n, c, p = logits.size()
        z = logits.float() * self.scale + self.tau * log_prior.float().view(1, c, 1)
        loss = F.cross_entropy(z, labels.unsqueeze(1).repeat(1, p), label_smoothing=self.eps)
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:   # accuracy of the raw (test-time) logits, mean over strips
            pred = logits.float().mean(-1).argmax(-1)
            self.info.update({'accuracy': (pred == labels).float().mean()})
        return loss, self.info
