import functools
import torch
import torch.nn as nn
import torch.nn.functional as F


class RDB(nn.Module):
    r"""
    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.2)
    """

    def __init__(self, channels: int = 64, grow_channels: int = 32, scale_ratio: float = 0.2):
        super(RDB, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(channels + 0 * grow_channels, grow_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(channels + 1 * grow_channels, grow_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(channels + 2 * grow_channels, grow_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(channels + 3 * grow_channels, grow_channels, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.conv5 = nn.Conv2d(channels + 4 * grow_channels, channels, kernel_size=3, stride=1, padding=1)

        self.scale_ratio = scale_ratio

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv1 = self.conv1(x)
        conv2 = self.conv2(torch.cat((x, conv1), dim=1))
        conv3 = self.conv3(torch.cat((x, conv1, conv2), dim=1))
        conv4 = self.conv4(torch.cat((x, conv1, conv2, conv3), dim=1))
        conv5 = self.conv5(torch.cat((x, conv1, conv2, conv3, conv4), dim=1))

        out = torch.add(conv5 * self.scale_ratio, x)

        return out


class RRDB(nn.Module):
    r"""
    Args:
        channels (int): Number of channels in the input image. (Default: 64)
        growth_channels (int): how many filters to add each layer (`k` in paper). (Default: 32)
        scale_ratio (float): Residual channel scaling column. (Default: 0.2)
    """

    def __init__(self, channels: int = 64, grow_channels: int = 32, scale_ratio: float = 0.2):
        super(RRDB, self).__init__()

        self.RDB1 = RDB(channels, grow_channels)
        self.RDB2 = RDB(channels, grow_channels)
        self.RDB2 = RDB(channels, grow_channels)

        self.scale_ratio = scale_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)

        out = torch.add(out * self.scale_ratio, x)

        return out
