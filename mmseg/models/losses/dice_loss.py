import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class DiceLoss(nn.Module):
    def __init__(self,
                 smooth=1.0,
                 exponent=2,
                 reduction='mean',
                 loss_weight=1.0,
                 ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.exponent = exponent
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.ignore_index = ignore_index

    def forward(self,
            pred,
            target,
            weight=None,
            ignore_index=None,
            **kwargs):


        """
        pred: (N, C, H, W)
        target: (N, H, W)
        """

        num_classes = pred.shape[1]
        pred = F.softmax(pred, dim=1)

        target = target.clone()
        valid_mask = target != self.ignore_index
        target[~valid_mask] = 0

        target_one_hot = F.one_hot(target, num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()

        pred = pred * valid_mask.unsqueeze(1)
        target_one_hot = target_one_hot * valid_mask.unsqueeze(1)

        dims = (0, 2, 3)
        intersection = torch.sum(pred * target_one_hot, dims)
        union = torch.sum(
            pred.pow(self.exponent) + target_one_hot.pow(self.exponent),
            dims
        )

        dice = (2 * intersection + self.smooth) / (union + self.smooth)
        loss = 1 - dice

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return self.loss_weight * loss
