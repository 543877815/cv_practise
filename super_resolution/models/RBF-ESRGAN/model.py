import torch

from .RRDB import RRDB
from .RRFDB import RRFDB, Upsampling, RFB
import torch.nn as nn


# https://github.com/Lornatang/RFB_ESRGAN-PyTorch/blob/master/rfb_esrgan_pytorch/models/generator.py
class Generator(nn.Module):
    def __init__(self, num_RRDB: int = 16, num_RRFDB: int = 8, img_channels: int = 3,
                 channels: int = 64, growth_channels: int = 64, scale_ratio: float = 0.1):
        super(Generator).__init__()

        # first layer
        self.conv1 = nn.Conv2d(img_channels, channels, kernel_size=3, stride=1, padding=1)

        # 16 RRDB
        self.Trunk_a = nn.ModuleList()
        for _ in range(num_RRDB):
            self.Trunk_a.append(RRDB(channels, growth_channels, scale_ratio))

        # 8 RFB-RDB
        self.Trunk_RFB = nn.ModuleList()
        for _ in range(num_RRFDB):
            self.Trunk_RFB.append(RRFDB(channels, channels, scale_ratio))

        # second conv layer post Trunk-RFB
        self.RFB = RFB(channels, channels, scale_ratio)

        # upsampling layer
        self.upsampling = Upsampling(channels)

        # Next conv layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        # Final output layer
        self.conv3 = nn.Conv2d(channels, img_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)

        trunk_a = self.Trunk_a(conv1)
        trunk_rbf = self.Trunk_RFB(trunk_a)

        out = torch.add(conv1, trunk_rbf)

        out = self.RFB(out)
        out = self.conv2(out)
        out = self.conv3(out)

        return out


# https://github.com/Lornatang/RFB_ESRGAN-PyTorch/blob/master/rfb_esrgan_pytorch/models/discriminator.py
class DiscriminatorForVGG(nn.Module):
    def __init__(self, image_size: int = 512) -> None:
        super(DiscriminatorForVGG, self).__init__()

        feature_map_size = image_size // 32

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),  # input is (3) x 512 x 512
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (64) x 256 x 256
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (128) x 128 x 128
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (256) x 64 x 64
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (512) x 32 x 32
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1, bias=False),  # state size. (512) x 16 x 16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear(512 * feature_map_size * feature_map_size, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out


def discriminator_for_vgg(image_size: int = 512) -> DiscriminatorForVGG:
    r"""GAN model architecture from `<https://arxiv.org/pdf/2005.12597.pdf>` paper."""
    model = DiscriminatorForVGG(image_size)
    return model
