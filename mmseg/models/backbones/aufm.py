import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint
import math

@BACKBONES.register_module()
class AUFM(nn.Module):
    """
    Adaptive Underwater Feature Modulation (AUFM)

    Learns a quality-aware attention map to enhance degraded underwater features.
    """

    def __init__(self, in_channels):
        super(AUFM, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels // 4)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels // 4, 1, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Learn quality map
        q = self.conv1(x)
        q = self.bn1(q)
        q = self.relu(q)

        q = self.conv2(q)
        q = self.sigmoid(q)   # [B,1,H,W]

        # Feature modulation
        out = x * q + x   # residual enhancement

        return out