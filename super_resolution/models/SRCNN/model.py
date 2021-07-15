# https://github.com/icpm/super-resolution
import torch
import torch.nn as nn
import torch.nn.functional as F


class SRCNN(torch.nn.Module):
    def __init__(self, img_channels=1, num_filter=64):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(img_channels, num_filter, kernel_size=9, padding=9 // 2, stride=1)
        self.conv2 = nn.Conv2d(num_filter, num_filter // 2, kernel_size=5, padding=5 // 2, stride=1)
        self.conv3 = nn.Conv2d(num_filter // 2, img_channels, kernel_size=5, padding=5 // 2, stride=1)
        self.relu = nn.ReLU(inplace=True)
        self.weight_init(mean=0.0, std=0.001)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

    # not good
    def weight_init(self, mean=0.0, std=0.001):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()
