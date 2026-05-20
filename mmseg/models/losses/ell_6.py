import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models.builder import LOSSES


@LOSSES.register_module()
class EdgeLearningLoss(nn.Module):
    def __init__(self, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight

        # Scharr kernels (fixed, no grad)
        kx = torch.tensor([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        ky = torch.tensor([
            [-3, -10, -3],
            [0, 0, 0],
            [3, 10, 3]
        ], dtype=torch.float32).view(1, 1, 3, 3)

        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(
        self,
        pred,
        target,
        weight=None,          # REQUIRED by MMSeg
        ignore_index=255      # REQUIRED by MMSeg
    ):
        """
        pred: (N, C, H, W) logits
        target: (N, 1, H, W) GT labels
        """

        # Convert multi-class prediction → binary foreground
        pred_prob = torch.softmax(pred, dim=1)
        pred_fg = 1.0 - pred_prob[:, 0:1, :, :]  # treat class-0 as background

        # Binary GT
        gt_fg = (target != 0).float()

        # Edge maps
        pred_edge = self._edge_map(pred_fg)
        gt_edge = self._edge_map(gt_fg)

        # Binary cross entropy on edges
        loss = F.binary_cross_entropy(pred_edge, gt_edge)

        return self.loss_weight * loss

    def _edge_map(self, x):
        gx = F.conv2d(x, self.kx, padding=1)
        gy = F.conv2d(x, self.ky, padding=1)
        edge = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
        edge = torch.clamp(edge, 0, 1)
        return edge
