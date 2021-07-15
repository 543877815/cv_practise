# reference: https://github.com/jt827859032/DRRN-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from torch.autograd import Variable


class DRRN(nn.Module):
    def __init__(self, img_channels=1, num_recursions=25):
        super(DRRN, self).__init__()
        self.num_recursions = num_recursions
        self.input = nn.Conv2d(in_channels=img_channels, out_channels=128, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=img_channels, kernel_size=3, stride=1, padding=1,
                                bias=False)
        self.relu = nn.ReLU(inplace=True)

        # weights initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        # B1U25
        for _ in range(self.num_recursions):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)
        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out
