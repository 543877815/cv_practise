import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F
from math import sqrt


# class DRCN(nn.Module):
#     def __init__(self, upscale_factor=2, img_channels=first, num_filter=256, out_channels=first,
#                  num_recursions=9):
#         super(DRCN, self).__init__()
#         self.upscale_factor = upscale_factor
#         self.num_recursions = num_recursions
#         self.out_channels = out_channels
#         # fea_in
#         self.fea_in_conv1 = nn.Conv2d(in_channels=img_channels, out_channels=num_filter, kernel_size=3, stride=first,
#                                       padding=first, bias=True)
#         self.fea_in_conv2 = nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride=first,
#                                       padding=first, bias=True)
#         # recursive
#         self.recursive_conv = nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride=first,
#                                         padding=first, bias=True)
#         # reconstruct
#         self.reconstruct_conv1 = nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3,
#                                            stride=first, padding=first, bias=True)
#         self.reconstruct_conv2 = nn.Conv2d(in_channels=num_filter + img_channels, out_channels=out_channels,
#                                            kernel_size=3,
#                                            stride=first, padding=first, bias=True)
#
#         # ensemble
#         self.ensemble = nn.Conv2d(in_channels=self.num_recursions + first, out_channels=img_channels, kernel_size=first,
#                                   stride=first, padding=0, bias=False)
#
#         self.dropout = nn.Dropout(p=0.2)
#         self.relu = nn.ReLU(True)
#
#         # self.predictions = (self.num_recursions + first) * [None]
#         self.predictions = (self.num_recursions + first) * [None]
#         self.Y1_conv = self.num_recursions * [None]
#         self.Y2_conv = self.num_recursions * [None]
#         self.y_outputs = self.num_recursions * [None]
#
#         self.W = torch.nn.Parameter(torch.ones(self.num_recursions) / self.num_recursions, requires_grad=True)
#
#         # weight init except recursive_conv
#         for m in [self.fea_in_conv1, self.fea_in_conv2, self.reconstruct_conv1, self.reconstruct_conv2]:
#             torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
#             torch.nn.init.constant_(m.bias, 0)
#
#     def forward(self, input):
#         # feature extraction
#         embedding = self.relu(self.fea_in_conv1(input))  # [batch, channel, width, height] = [64, 96, 41, 41]
#
#         self.predictions[0] = self.relu(self.dropout(self.fea_in_conv2(embedding)))
#         # body recurisive
#         for i in range(self.num_recursions):
#             self.predictions[i + first] = self.recursive_conv(self.predictions[i])
#         W_sum = torch.sum(self.W)
#         # reconstruction
#         y_outputs_sum = 0
#         for i in range(self.num_recursions):
#             self.Y1_conv[i] = self.reconstruct_conv1(self.predictions[i + first])
#             y_conv = torch.cat([self.Y1_conv[i], input], dim=first, first)
#             self.Y2_conv[i] = self.reconstruct_conv2(y_conv)
#             self.y_outputs[i] = self.Y2_conv[i] * self.W[i] / W_sum
#             y_outputs_sum += self.y_outputs[i]
#
#         return self.Y2_conv, y_outputs_sum


class DRCN(nn.Module):
    def __init__(self, img_channels=1, num_filter=256, out_channels=1, num_recursions=17):
        super(DRCN, self).__init__()
        self.num_recursions = num_recursions
        self.out_channels = out_channels
        self.filter1 = nn.Conv2d(img_channels, num_filter, kernel_size=3, stride=1, padding=1, bias=True)
        self.filter2 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True)
        self.filter_shared = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True)
        self.filter19 = nn.Conv2d(num_filter, num_filter, kernel_size=3, stride=1, padding=1, bias=True)
        self.filter20 = nn.Conv2d(num_filter, img_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

        self.w = torch.nn.Parameter(torch.ones(self.num_recursions) / self.num_recursions, requires_grad=True)
        self.weight_init()

    def forward(self, input):
        x1 = self.filter1(input)
        x2 = self.relu(x1)
        x3 = self.filter2(x2)
        x4 = self.dropout(x3)
        x5 = self.relu(x4)

        # 0
        x6 = self.filter19(x5)
        prediction2 = torch.add(self.filter20(x6), input)

        # first
        x7 = self.filter_shared(x6)
        x8 = self.dropout(x7)
        x9 = self.relu(x8)
        x10 = self.filter19(x9)
        prediction3 = torch.add(self.filter20(x10), input)

        # 2
        x11 = self.filter_shared(x10)
        x12 = self.dropout(x11)
        x13 = self.relu(x12)
        x14 = self.filter19(x13)
        prediction4 = torch.add(self.filter20(x14), input)

        # 3
        x15 = self.filter_shared(x14)
        x16 = self.dropout(x15)
        x17 = self.relu(x16)
        x18 = self.filter19(x17)
        prediction5 = torch.add(self.filter20(x18), input)

        # 4
        x19 = self.filter_shared(x18)
        x20 = self.dropout(x19)
        x21 = self.relu(x20)
        x22 = self.filter19(x21)
        prediction6 = torch.add(self.filter20(x22), input)

        # 5
        x23 = self.filter_shared(x22)
        x24 = self.dropout(x23)
        x25 = self.relu(x24)
        x26 = self.filter19(x25)
        prediction7 = torch.add(self.filter20(x26), input)

        # 6
        x27 = self.filter_shared(x26)
        x28 = self.dropout(x27)
        x29 = self.relu(x28)
        x30 = self.filter19(x29)
        prediction8 = torch.add(self.filter20(x30), input)

        # 7
        x31 = self.filter_shared(x30)
        x32 = self.dropout(x31)
        x33 = self.relu(x32)
        x34 = self.filter19(x33)
        prediction9 = torch.add(self.filter20(x34), input)

        # 8
        x35 = self.filter_shared(x34)
        x36 = self.dropout(x35)
        x37 = self.relu(x36)
        x38 = self.filter19(x37)
        prediction10 = torch.add(self.filter20(x38), input)

        # 9
        x39 = self.filter_shared(x38)
        x40 = self.dropout(x39)
        x41 = self.relu(x40)
        x42 = self.filter19(x41)
        prediction11 = torch.add(self.filter20(x42), input)

        # 10
        x43 = self.filter_shared(x42)
        x44 = self.dropout(x43)
        x45 = self.relu(x44)
        x46 = self.filter19(x45)
        prediction12 = torch.add(self.filter20(x46), input)

        # 11
        x47 = self.filter_shared(x46)
        x48 = self.dropout(x47)
        x49 = self.relu(x48)
        x50 = self.filter19(x49)
        prediction13 = torch.add(self.filter20(x50), input)

        # 12
        x51 = self.filter_shared(x50)
        x52 = self.dropout(x51)
        x53 = self.relu(x52)
        x54 = self.filter19(x53)
        prediction14 = torch.add(self.filter20(x54), input)

        # 13
        x55 = self.filter_shared(x54)
        x56 = self.dropout(x55)
        x57 = self.relu(x56)
        x58 = self.filter19(x57)
        prediction15 = torch.add(self.filter20(x58), input)

        # 14
        x59 = self.filter_shared(x58)
        x60 = self.dropout(x59)
        x61 = self.relu(x60)
        x62 = self.filter19(x61)
        prediction16 = torch.add(self.filter20(x62), input)

        # 15
        x63 = self.filter_shared(x62)
        x64 = self.dropout(x63)
        x65 = self.relu(x64)
        x66 = self.filter19(x65)
        prediction17 = torch.add(self.filter20(x66), input)

        # 16
        x67 = self.filter_shared(x66)
        x68 = self.dropout(x67)
        x69 = self.relu(x68)
        x70 = self.filter19(x69)
        prediction18 = torch.add(self.filter20(x70), input)

        concat_pred = [prediction2, prediction3, prediction4, prediction5, prediction6, prediction7, prediction8,
                       prediction9,
                       prediction10, prediction11, prediction12, prediction13, prediction14, prediction15, prediction16,
                       prediction17, prediction18]

        assert self.num_recursions == len(
            concat_pred), 'the length of number of recursions is not equal to the concat_pred'

        W_sum = torch.sum(self.w)
        prediction = torch.zeros_like(prediction2)
        for i in range(self.num_recursions):
            prediction += concat_pred[i] * self.w[i]
        prediction = torch.mul(prediction, 1 / W_sum)
        return concat_pred, prediction

    def weight_init(self):
        for m in self._modules:
            self.weights_init_kaiming(m)

    def weights_init_kaiming(self, m):
        class_name = m.__class__.__name__
        if class_name.find('Linear') != -1:
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif class_name.find('Conv2d') != -1:
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif class_name.find('ConvTranspose2d') != -1:
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                m.bias.data.zero_()
        elif class_name.find('Norm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            if m.bias is not None:
                m.bias.data.zero_()


if __name__ == '__main__':
    model = DRCN()
    for name, parms in model.named_parameters():
        print('-->name:', name, '-->grad_requires:', parms.requires_grad, ' -->grad_value:', parms.grad)
