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
        self.fea_in_conv = nn.Conv2d(in_channels=num_channels, out_channels=base_channel, kernel_size=3, stride=1,
                                     padding=1, bias=True)
        self.fea_in_ReLU = nn.ReLU(True)
        # fea_in init
        torch.nn.init.kaiming_normal_(self.fea_in_conv.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.fea_in_conv.bias, 0)

        # recursive
        self.recursive_conv1 = nn.Conv2d(in_channels=base_channel, out_channels=base_channel, kernel_size=3, stride=1,
                                         padding=1, bias=True)
        self.recursive_Dropout = nn.Dropout(p=0.2)
        self.recursive_ReLU = nn.ReLU(True)
        self.recursive_conv2 = nn.Conv2d(in_channels=base_channel, out_channels=base_channel, kernel_size=3, stride=1,
                                         padding=1, bias=True)
        # recursive init
        torch.nn.init.kaiming_normal_(self.recursive_conv1.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.recursive_conv1.bias, 0)

        # reconstruct
        self.reconstruct_conv_1 = nn.Conv2d(in_channels=base_channel, out_channels=num_channels, kernel_size=3,
                                            stride=1, padding=1, bias=True)
        self.reconstruct_conv_2 = nn.Conv2d(in_channels=num_recursions, out_channels=out_channels, kernel_size=1,
                                            stride=1, padding=0, bias=True)
        # reconstruct init
        torch.nn.init.kaiming_normal_(self.reconstruct_conv_1.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.reconstruct_conv_1.bias, 0)
        torch.nn.init.kaiming_normal_(self.reconstruct_conv_2.weight, nonlinearity='relu')
        torch.nn.init.constant_(self.reconstruct_conv_2.bias, 0)

        # weight init for recursions
        self.adjust_weight = torch.nn.Parameter(torch.ones(self.num_recursions) / self.num_recursions,
                                                requires_grad=True).to(device)

        self.adjust_weight_sum = torch.sum(self.adjust_weight).to(device)

    def forward(self, x):
        # feature extraction
        h = self.fea_in_conv(x)
        h = self.fea_in_ReLU(h)

        # body recurisive
        yd = [torch.empty(x.shape)] * (self.num_recursions)
        for i in range(self.num_recursions):
            h = self.recursive_conv1(h)
            h = self.recursive_Dropout(h)
            h = self.recursive_ReLU(h)
            h = self.recursive_conv2(h)

            yd[i] = torch.add(self.reconstruct_conv_1(h), x)  # prediction for recursive

        y_recon = torch.cat(yd, 1)  # input for reconstruction

        # reconstruction
        output = self.reconstruct_conv_2(y_recon)
        return yd, torch.add(output, x)


# class DRCN(nn.Module):
#     def __init__(self, device, upscale_factor=2, num_channels=1, base_channel=256, out_channels=1, num_recursions=9):
#         super(DRCN, self).__init__()
#         self.upscale_factor = upscale_factor
#         self.num_recursions = num_recursions
#         self.device = device
#
#         # feature extraction
#         self.fea_in = nn.Sequential(
#             nn.Conv2d(in_channels=num_channels, out_channels=base_channel, kernel_size=3, stride=1, padding=1,
#                       bias=True),
#             nn.ReLU(True),
#             nn.Conv2d(in_channels=base_channel, out_channels=base_channel, kernel_size=3, stride=1, padding=1,
#                       bias=True),
#             nn.ReLU(True)
#         )
#
#         # body recursive
#         self.recursive = nn.Sequential(
#             nn.Conv2d(in_channels=base_channel, out_channels=base_channel, kernel_size=3, stride=1, padding=1,
#                       bias=True),
#             nn.ReLU(True)
#         )
#
#         # reconstruction
#         self.reconstruct_conv1 = nn.Sequential(
#             nn.Conv2d(in_channels=base_channel, out_channels=base_channel, kernel_size=3, stride=1, padding=1,
#                       bias=True),
#             nn.ReLU(True),
#         )
#
#         self.reconstruct_conv2 = nn.Sequential(
#             nn.Conv2d(in_channels=base_channel + num_channels, out_channels=out_channels, kernel_size=3, stride=1,
#                       padding=1,
#                       bias=True),
#             nn.ReLU(True)
#         )
#
#         # weight init for recursions
#         self.adjust_weight = torch.nn.Parameter(torch.ones(self.num_recursions) / self.num_recursions,
#                                                 requires_grad=True).to(device)
#
#         self.adjust_weight_sum = torch.sum(self.adjust_weight).to(device)
#
#         # weights initialization
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 torch.nn.init.kaiming_normal_(m.weight)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#                 # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 # m.weight.data.normal_(0, sqrt(2. / n))
#
#     def forward(self, x):
#         # feature extraction
#         out = self.fea_in(x)
#
#         # body recurisive
#         yd = [torch.empty(x.shape)] * (self.num_recursions + 1)
#         h = out
#         yd[0] = out
#         for i in range(self.num_recursions):
#             h = self.recursive(h)
#             yd[i + 1] = h
#
#         # reconstruction
#         Y1_conv = [torch.empty(x.shape)] * self.num_recursions
#         Y2_conv = [torch.empty(x.shape)] * self.num_recursions
#         out_sum = torch.zeros_like(x)
#         for i in range(self.num_recursions):
#             recon1 = self.reconstruct_conv1(yd[i + 1])
#             Y1_conv[i] = recon1
#             y_conv = torch.cat((Y1_conv[i], x), 1)  # 256 + 1
#             Y2_conv[i] = self.reconstruct_conv2(y_conv)
#             out_sum += Y2_conv[i] * self.adjust_weight[i] / self.adjust_weight_sum
#
#         for i in range(self.num_recursions):
#             Y2_conv[i] = Y2_conv[i] + x
#
#         return Y2_conv, torch.add(out_sum, x)


if __name__ == '__main__':
    model = DRCN()
    for name, parms in model.named_parameters():
        print('-->name:', name, '-->grad_requires:', parms.requires_grad, ' -->grad_value:', parms.grad)
