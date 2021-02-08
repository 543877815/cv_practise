# https://github.com/icpm/super-resolution
import torch
import torch.nn as nn


class SRCNN(torch.nn.Module):
    def __init__(self, num_channels, filter, params=None):
        super(SRCNN, self).__init__()

        if params is None:
            params = [9, 5, 5]
        if len(params) == 3:
            self.layers = torch.nn.Sequential(
                nn.Conv2d(in_channels=num_channels, out_channels=filter, kernel_size=params[0], padding=params[0] // 2,
                          stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=filter, out_channels=filter // 2, kernel_size=params[1], padding=params[1] // 2,
                          stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=filter // 2, out_channels=num_channels, kernel_size=params[2], stride=1,
                          padding=params[2] // 2, bias=True),
            )
        elif len(params) == 4:
            self.layers = torch.nn.Sequential(
                nn.Conv2d(in_channels=num_channels, out_channels=filter, kernel_size=params[0], padding=params[0] // 2,
                          stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=filter, out_channels=filter // 2, kernel_size=params[1], padding=params[1] // 2,
                          stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=filter, out_channels=filter // 2, kernel_size=params[2], padding=params[2] // 2,
                          stride=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=filter // 2, out_channels=num_channels, kernel_size=params[3], stride=1,
                          padding=params[3] // 2, bias=True),
            )

    def forward(self, x):
        out = self.layers(x)
        return out

    def weight_init(self, mean, std):
        for m in self._modules:
            self.normal_init(self._modules[m], mean, std)

    @staticmethod
    def normal_init(m, mean, std):
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            m.weight.data.normal_(mean, std)
            m.bias.data.zero_()
