import torch
import torch.nn as nn
import torch.nn.functional as F

class TwoWayLoss(nn.Module):
    """
    Two-way loss for multi-label classification.

    Reference:
    T. Kobayashi, 
    "Two-Way Multi-Label Loss," 
    2023 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), Vancouver, BC, Canada, 2023, pp. 7476-7485, 
    doi: 10.1109/CVPR52729.2023.00722.
    """
    def __init__(self, Tp=1., Tn=1.):
        super(TwoWayLoss, self).__init__()
        self.Tp = Tp
        self.Tn = Tn
        self.register_buffer('nINF', torch.tensor(-1e4))

    def forward(self, x, y):
        class_mask = (y > 0).any(dim=0)
        sample_mask = (y > 0).any(dim=1)

        pmask = y.masked_fill(y <= 0, self.nINF).masked_fill(y > 0, 0.0)
        plogit_class = torch.logsumexp(-x / self.Tp + pmask, dim=0).mul(self.Tp)[class_mask]
        plogit_sample = torch.logsumexp(-x / self.Tp + pmask, dim=1).mul(self.Tp)[sample_mask]

        nmask = y.masked_fill(y != 0, self.nINF).masked_fill(y == 0, 0.0)
        nlogit_class = torch.logsumexp(x / self.Tn + nmask, dim=0).mul(self.Tn)[class_mask]
        nlogit_sample = torch.logsumexp(x / self.Tn + nmask, dim=1).mul(self.Tn)[sample_mask]

        loss = 0
        if plogit_class.numel() > 0 and nlogit_class.numel() > 0:
             loss += F.softplus(nlogit_class + plogit_class).mean()
        if plogit_sample.numel() > 0 and nlogit_sample.numel() > 0:
             loss += F.softplus(nlogit_sample + plogit_sample).mean()
        return loss

def get_loss(loss_type):
    if loss_type == "twoway":
        return TwoWayLoss()
    elif loss_type == "bce":
        # classic binary cross entropy with logits
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
