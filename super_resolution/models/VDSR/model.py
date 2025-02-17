import torch
import torch.nn as nn
from math import sqrt
from torch.nn import functional as F


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x))


class VDSR(nn.Module):
    def __init__(self, img_channels, num_residuals=18, num_filter=64):
        super(VDSR, self).__init__()
        self.input = nn.Conv2d(in_channels=img_channels, out_channels=num_filter, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.residual_layer = self.make_layer(Conv_ReLU_Block, num_residuals)
        self.output = nn.Conv2d(in_channels=num_filter, out_channels=1, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def make_layer(self, block, num_of_layer):
        layers = []
        for _ in range(num_of_layer):
            layers.append(block())
        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        out = self.residual_layer(out)
        out = self.output(out)
        out = torch.add(out, residual)
        return out
