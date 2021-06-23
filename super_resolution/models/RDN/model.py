import torch
from torch import nn


class DenseLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DenseLayer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=3 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return torch.cat([x, self.relu(self.conv(x))], 1)


class RDB(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(RDB, self).__init__()
        self.layers = nn.Sequential(
            *[DenseLayer(in_channels + growth_rate * i, growth_rate) for i in range(num_layers)])

        # local feature fusion
        self.lff = nn.Conv2d(in_channels + growth_rate * num_layers, growth_rate, kernel_size=1)

    def forward(self, x):
        return x + self.lff(self.layers(x))  # local residual learning


class RDN(nn.Module):
    def __init__(self, scale_factor=2, num_channels=43, num_features=64, growth_rate=64, num_blocks=16, num_layers=8):
        super(RDN, self).__init__()
        self.G0 = num_features  # 64
        self.G = growth_rate  # 64
        self.D = num_blocks  # 16
        self.C = num_layers  # 8

        # shallow feature extraction
        self.sfe1 = nn.Conv2d(num_channels, num_features, kernel_size=3, padding=3 // 2)
        self.sfe2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=3 // 2)

        # residual dense blocks
        self.rdbs = nn.ModuleList([RDB(self.G0, self.G, self.C)])
        for _ in range(self.D - 1):
            self.rdbs.append(RDB(self.G, self.G, self.C))

        # global feature fusion
        self.gff = nn.Sequential(
            nn.Conv2d(self.G * self.D, self.G0, kernel_size=1),
            nn.Conv2d(self.G0, self.G0, kernel_size=3, padding=3 // 2)
        )

        # up-sampling
        assert 2 <= scale_factor <= 4
        if scale_factor == 2 or scale_factor == 4:
            self.upscale = []
            for _ in range(scale_factor // 2):
                self.upscale.extend([nn.Conv2d(self.G0, self.G0 * (2 ** 2), kernel_size=3, padding=3 // 2),
                                     nn.PixelShuffle(2)])
            self.upscale = nn.Sequential(*self.upscale)
        else:
            self.upscale = nn.Sequential(
                nn.Conv2d(self.G0, self.G0 * (scale_factor ** 2), kernel_size=3, padding=3 // 2),
                nn.PixelShuffle(scale_factor)
            )

        self.output = nn.Conv2d(self.G0, num_channels, kernel_size=3, padding=3 // 2)

    def forward(self, x):
        sfe1 = self.sfe1(x)
        sfe2 = self.sfe2(sfe1)

        x = sfe2
        local_features = []
        for i in range(self.D):
            x = self.rdbs[i](x)
            local_features.append(x)

        x = self.gff(torch.cat(local_features, 1)) + sfe1  # global residual learning
        x = self.upscale(x)
        x = self.output(x)
        return x

# import torch
# import torch.nn as nn
#
# def make_model(args, parent=False):
#     return RDN(args)
#
#
# class RDB_Conv(nn.Module):
#     def __init__(self, inChannels, growRate, kSize=3):
#         super(RDB_Conv, self).__init__()
#         Cin = inChannels
#         G = growRate
#         self.conv = nn.Sequential(*[
#             nn.Conv2d(Cin, G, kSize, padding=(kSize - 1) // 2, stride=1),
#             nn.ReLU()
#         ])
#
#     def forward(self, x):
#         out = self.conv(x)
#         return torch.cat((x, out), 1)
#
#
# class RDB(nn.Module):
#     def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
#         super(RDB, self).__init__()
#         G0 = growRate0
#         G = growRate
#         C = nConvLayers
#
#         convs = []
#         for c in range(C):
#             convs.append(RDB_Conv(G0 + c * G, G))
#         self.convs = nn.Sequential(*convs)
#
#         # Local Feature Fusion
#         self.LFF = nn.Conv2d(G0 + C * G, G0, 1, padding=0, stride=1)
#
#     def forward(self, x):
#         return self.LFF(self.convs(x)) + x
#
#
# class RDN(nn.Module):
#     def __init__(self, r, args):
#         super(RDN, self).__init__()
#         r = r
#         G0 = 64
#         kSize = 3
#
#         # number of RDB blocks, conv layers, out channels
#         self.D, C, G = {
#             'A': (20, 6, 32),
#             'B': (16, 8, 64),
#         }['B']
#
#         # Shallow feature extraction net
#         self.SFENet1 = nn.Conv2d(3, G0, kSize, padding=(kSize - 1) // 2, stride=1)
#         self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
#
#         # Redidual dense blocks and dense feature fusion
#         self.RDBs = nn.ModuleList()
#         for i in range(self.D):
#             self.RDBs.append(
#                 RDB(growRate0=G0, growRate=G, nConvLayers=C)
#             )
#
#         # Global Feature Fusion
#         self.GFF = nn.Sequential(*[
#             nn.Conv2d(self.D * G0, G0, 1, padding=0, stride=1),
#             nn.Conv2d(G0, G0, kSize, padding=(kSize - 1) // 2, stride=1)
#         ])
#
#         # Up-sampling net
#         if r == 2 or r == 3:
#             self.UPNet = nn.Sequential(*[
#                 nn.Conv2d(G0, G * r * r, kSize, padding=(kSize - 1) // 2, stride=1),
#                 nn.PixelShuffle(r),
#                 nn.Conv2d(G, 3, kSize, padding=(kSize - 1) // 2, stride=1)
#             ])
#         elif r == 4:
#             self.UPNet = nn.Sequential(*[
#                 nn.Conv2d(G0, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
#                 nn.PixelShuffle(2),
#                 nn.Conv2d(G, G * 4, kSize, padding=(kSize - 1) // 2, stride=1),
#                 nn.PixelShuffle(2),
#                 nn.Conv2d(G, 3, kSize, padding=(kSize - 1) // 2, stride=1)
#             ])
#         else:
#             raise ValueError("scale must be 2 or 3 or 4.")
#
#     def forward(self, x):
#         f__1 = self.SFENet1(x)
#         x = self.SFENet2(f__1)
#
#         RDBs_out = []
#         for i in range(self.D):
#             x = self.RDBs[i](x)
#             RDBs_out.append(x)
#
#         x = self.GFF(torch.cat(RDBs_out, 1))
#         x += f__1
#
#         return self.UPNet(x)