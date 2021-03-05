import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class DRCN(nn.Module):
    def __init__(self, device, upscale_factor=2, num_channels=1, base_channel=256, out_channels=1, num_recursions=9):
        super(DRCN, self).__init__()
        self.upscale_factor = upscale_factor
        self.num_recursions = num_recursions
        self.device = device
        # fea_in
        self.fea_in_conv1 = nn.Conv2d(in_channels=num_channels, out_channels=base_channel, kernel_size=3, stride=1,
                                      padding=1, bias=True)
        self.fea_in_conv2 = nn.Conv2d(in_channels=base_channel, out_channels=base_channel, kernel_size=3, stride=1,
                                      padding=1, bias=True)
        # recursive
        self.recursive_conv = nn.Conv2d(in_channels=base_channel, out_channels=base_channel, kernel_size=3, stride=1,
                                        padding=1, bias=True)
        # reconstruct
        self.reconstruct_conv1 = nn.Conv2d(in_channels=base_channel, out_channels=base_channel, kernel_size=3,
                                           stride=1, padding=1, bias=True)
        self.reconstruct_conv2 = nn.Conv2d(in_channels=base_channel, out_channels=out_channels, kernel_size=1,
                                           stride=1, padding=0, bias=True)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(True)

        # weight init except recursive_conv
        for m in [self.fea_in_conv1, self.fea_in_conv2, self.reconstruct_conv1, self.reconstruct_conv2]:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            torch.nn.init.constant_(m.bias, 0)

        # weight init for recursions
        self.adjust_weight = torch.nn.Parameter(torch.ones(self.num_recursions + 1) / (self.num_recursions + 1),
                                                requires_grad=True).to(self.device)

    def forward(self, x):
        residual = x
        # feature extraction
        x = self.relu(self.fea_in_conv1(x))
        x = self.relu(self.dropout(self.fea_in_conv2(x)))
        h = self.reconstruct_conv1(x)
        y0 = torch.add(self.reconstruct_conv2(h), residual)
        yd = [y0]

        output = y0 * self.adjust_weight[0]

        # body recurisive
        for i in range(self.num_recursions):
            h = self.relu(self.dropout(self.recursive_conv(h)))

            # reconstruction
            h = self.reconstruct_conv1(h)
            yd.append(torch.add(self.reconstruct_conv2(h), residual))
            output += yd[i + 1] * self.adjust_weight[i + 1]

        return yd, output


if __name__ == '__main__':
    model = DRCN()
    for name, parms in model.named_parameters():
        print('-->name:', name, '-->grad_requires:', parms.requires_grad, ' -->grad_value:', parms.grad)
