# paper: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network（CVPR)
# url: https://arxiv.org/abs/1609.05158
# reference: https://github.com/yjn870/ESPCN-pytorch

import math
from torch import nn


class ESPCN(nn.Module):
    def __init__(self, upscale_factor, num_channels=1, filter=64):
        super(ESPCN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2d(num_channels, filter, kernel_size=5, padding=5 // 2),
            nn.Tanh(),
            nn.Conv2d(filter, filter // 2, kernel_size=3, padding=3 // 2),
            nn.Tanh(),
        )
        self.last_part = nn.Sequential(
            nn.Conv2d(filter // 2, num_channels * (upscale_factor ** 2), kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(upscale_factor)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m.in_channels == 32:
                    nn.init.normal_(m.weight.data, mean=0.0, std=0.001)
                    nn.init.zeros_(m.bias.data)
                else:
                    nn.init.normal_(m.weight.data, mean=0.0,
                                    std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                    nn.init.zeros_(m.bias.data)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x
