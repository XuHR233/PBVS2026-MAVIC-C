#utils/utils_reg.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        """
        alpha: Tensor of shape [num_classes], 类别权重
        gamma: focusing parameter
        reduction: 'mean' or 'sum' or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        logits: [B, num_classes]
        targets: [B] long tensor
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # pt = softmax概率中正确类别的概率

        if self.alpha is not None:
            at = self.alpha[targets]
            loss = at * (1 - pt) ** self.gamma * ce_loss
        else:
            loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss



def sliced_wasserstein_distance(x, y, num_projections=128):
    if x.dim() > 2:
        x = x.view(x.size(0), -1)
    if y.dim() > 2:
        y = y.view(y.size(0), -1)

    device = x.device
    dim = x.size(1)

    projections = torch.randn(num_projections, dim, device=device)
    projections = projections / torch.norm(projections, dim=1, keepdim=True)

    x_proj = x @ projections.T
    y_proj = y @ projections.T

    x_proj_sorted, _ = torch.sort(x_proj, dim=0)
    y_proj_sorted, _ = torch.sort(y_proj, dim=0)

    return torch.mean(torch.abs(x_proj_sorted - y_proj_sorted))


class da_loss(nn.Module):
    def __init__(self, num_projections=128):
        super().__init__()
        self.num_projections = num_projections

    def forward(self, feat_s, feat_t):
        return sliced_wasserstein_distance(feat_s, feat_t, self.num_projections)