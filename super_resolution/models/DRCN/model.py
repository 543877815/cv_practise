import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt


class DRCN(nn.Module):
    def __init__(self, upscale_factor=2, num_channels=1, num_filter=256, out_channels=1,
                 num_recursions=9):
        super(DRCN, self).__init__()
        self.upscale_factor = upscale_factor
        self.num_recursions = num_recursions
        # fea_in
        self.fea_in_conv1 = nn.Conv2d(in_channels=num_channels, out_channels=num_filter, kernel_size=3, stride=1,
                                      padding=1, bias=True)
        self.fea_in_conv2 = nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride=1,
                                      padding=1, bias=True)
        # recursive
        self.recursive_conv = nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3, stride=1,
                                        padding=1, bias=True)
        # reconstruct
        self.reconstruct_conv1 = nn.Conv2d(in_channels=num_filter, out_channels=num_filter, kernel_size=3,
                                           stride=1, padding=1, bias=True)
        self.reconstruct_conv2 = nn.Conv2d(in_channels=num_filter, out_channels=out_channels, kernel_size=3,
                                           stride=1, padding=1, bias=True)

        # ensemble
        self.ensemble = nn.Conv2d(in_channels=self.num_recursions + 1, out_channels=num_channels, kernel_size=1,
                                  stride=1, padding=0, bias=False)

        self.dropout = nn.Dropout(p=0.2)
        self.relu = nn.ReLU(True)

        # weight init except recursive_conv
        for m in [self.fea_in_conv1, self.fea_in_conv2, self.reconstruct_conv1, self.reconstruct_conv2]:
            torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, input):
        residual = input
        # feature extraction
        x2 = self.relu(self.fea_in_conv1(input))  # x2
        x5 = self.relu(self.dropout(self.fea_in_conv2(x2)))  # x5
        x6 = self.reconstruct_conv1(x5)  # x6

        prediction2 = torch.add(self.reconstruct_conv2(x6), residual)  # prediction2  -- torch.add
        predictions = prediction2

        # body recurisive
        for i in range(self.num_recursions):
            x9 = self.relu(self.dropout(self.dropout(self.recursive_conv(x6))))  # x9

            # reconstruction
            x10 = self.reconstruct_conv1(x9)  # x10

            prediction_i = torch.add(self.reconstruct_conv2(x10), residual)
            predictions = torch.cat((predictions, prediction_i), 1)
            x6 = x10
        return predictions, self.ensemble(predictions)


if __name__ == '__main__':
    model = DRCN()
    for name, parms in model.named_parameters():
        print('-->name:', name, '-->grad_requires:', parms.requires_grad, ' -->grad_value:', parms.grad)
