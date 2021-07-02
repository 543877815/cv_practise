# https://github.com/Lornatang/RFB_ESRGAN-PyTorch/blob/3a5e10936f4dc0ef44c042d3cb7ed09783a2fb6b/rfb_esrgan_pytorch/models/utils.py#L238
import torch
import torch.nn as nn


class RFB(nn.Module):
    r"""
    Args:
        in_channels (int): Number of channels in the input image. (Default: 64)
        out_channels (int): Number of channels produced by the convolution. (Default: 64)
        scale_ratio (float): Residual channel scaling column. (Default: 0.1)
    """

    def __init__(self, in_channels: int = 64, out_channels: int = 64, scale_ratio: float = 0.1):
        super(RFB, self).__init__()
        branch_channels = in_channels // 4

        # left 1
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=1, dilation=1)
        )
        # left 2
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        )
        # left 3
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3)
        )
        # left 4
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels // 2, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels // 2, (branch_channels // 4) * 3, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            # here different from the paper
            nn.Conv2d((branch_channels // 4) * 3, branch_channels, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(branch_channels, branch_channels, kernel_size=3, stride=1, padding=5, dilation=5)
        )

        # left 5 shortcut layer
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

        self.conv_linear = nn.Conv2d(4 * branch_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.scale_ratio = scale_ratio

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(x)
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        out = torch.cat((branch1, branch2, branch3, branch4), dim=1)
        out = self.conv_linear(out)

        out = self.leaky_relu(torch.add(out * self.scale_ratio, shortcut))

        return out


class RFDB(nn.Module):
    r""" Inspired by the multi-scale kernels and the structure of Receptive Fields (RFs) in human visual systems,
            RFB-SSD proposed Receptive Fields Block (RFB) for object detection
        Args:
            channels (int): Number of channels in the input image. (Default: 64)
            growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
            scale_ratio (float): Residual channel scaling column. (Default: 0.1)
    """

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.1):
        super(RFDB, self).__init__()

        self.rfb1 = nn.Sequential(
            RFB(channels + 0 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb2 = nn.Sequential(
            RFB(channels + 1 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb3 = nn.Sequential(
            RFB(channels + 2 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb4 = nn.Sequential(
            RFB(channels + 3 * growth_channels, growth_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.rfb5 = nn.Sequential(
            RFB(channels + 4 * growth_channels, growth_channels),
        )

        self.scale_ratio = scale_ratio

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rfb1 = self.rfb1(x)
        rfb2 = self.rfb2(torch.cat((x, rfb1), dim=1))
        rfb3 = self.rfb3(torch.cat((x, rfb1, rfb2), dim=1))
        rfb4 = self.rfb4(torch.cat((x, rfb1, rfb2, rfb3), dim=1))
        rfb5 = self.rfb5(torch.cat((x, rfb1, rfb2, rfb3, rfb4), dim=1))

        out = torch.add(rfb5 * self.scale_ratio, x)

        return out


class RRFDB(nn.Module):
    r"""The residual block structure of traditional RFB-ESRGAN is defined
    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.1)
    """

    def __init__(self, channels: int = 64, growth_channels: int = 32, scale_ratio: float = 0.1):
        super(RRFDB, self).__init__()
        self.RFDB1 = RFDB(channels, growth_channels)
        self.RFDB2 = RFDB(channels, growth_channels)
        self.RFDB3 = RFDB(channels, growth_channels)

        self.scale_ratio = scale_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RFDB1(x)
        out = self.RFDB2(out)
        out = self.RFDB3(out)

        out = torch.add(out * self.scale_ratio, x)
        return out


class Upsampling(nn.Module):
    def __init__(self, channels: int = 64):
        super(Upsampling).__init__()

        self.module1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearset'),
            RFB(channels, channels * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.module2 = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            RFB(channels, channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.module3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            RFB(channels, channels * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.module4 = nn.Sequential(
            nn.PixelShuffle(upscale_factor=2),
            RFB(channels, channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.module5 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        out = self.module1(x)
        out = self.module2(out)
        out = self.module3(out)
        out = self.module4(out)
        out = self.module5(out)

        return out
