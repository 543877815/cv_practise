import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
from .RRDB import RRDBNet
from .stylegan import G_synthesis


class GLEAN(nn.Module):
    def __init__(self, img_size, device=None,
                 channels: int = 64,
                 resolution: int = 1024,
                 in_nc: int = 3,
                 out_nc: int = 3):
        super().__init__()
        self.RRDB = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32)
        self.synthesis = G_synthesis().to(device)

        # 1024->512
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        # 512->256
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        # 256->128
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        # 128->64
        self.conv4 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        # 64->32
        self.conv5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        # 32->16
        self.conv6 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        # 16->8
        self.conv7 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        # 8->4
        self.conv8 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        out = self.RRDB(x)
