import functools
import torch
import torch.nn as nn
from super_resolution.models.GLEAN.RRDB import RRDBNet
from super_resolution.models.GLEAN.styleGAN import G_synthesis, G_mapping



class GLEAN(nn.Module):
    def __init__(self,
                 device=None,
                 channels: int = 64,
                 img_channels=3,
                 in_nc: int = 3,
                 nf: int = 64,
                 gc: int = 32,
                 nb: int = 23,
                 out_nc: int = 3):
        super().__init__()
        self.device = device

        # load pretrained model

        self.lrelu = torch.nn.LeakyReLU(negative_slope=0.2)

        # =======================================
        # ||              Encoder              ||
        # =======================================
        # 32->32
        self.E0 = RRDBNet(in_nc=in_nc, out_nc=out_nc, nf=nf, nb=nb, gc=gc)

        # 32->16
        self.E1 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        # 16->8
        self.E2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        # 8->4
        self.E3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

        # 4 * 4 * 64 -> 512
        self.E4_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.E4_fc = nn.Linear(4 * 4 * channels, 512)

        # =======================================
        # ||            MemoryBank             ||
        # =======================================
        self.merge1 = nn.Conv2d(channels + 512, 512, kernel_size=3, stride=1,
                                padding=1)  # (512 + 64, 4, 4) -> (512 + 512, 4, 4)
        self.merge2 = nn.Conv2d(channels + 512, 512, kernel_size=3, stride=1,
                                padding=1)  # (512 + 64, 8, 8) -> (512 + 512, 8, 8)
        self.merge3 = nn.Conv2d(channels + 512, 512, kernel_size=3, stride=1,
                                padding=1)  # (512 + 64, 16, 16) -> (512 + 512, 16, 16)
        self.merge4 = nn.Conv2d(channels + 512, 512, kernel_size=3, stride=1,
                                padding=1)  # (512 + 64, 32, 32) -> (512 + 512, 32, 32)

        # =======================================
        # ||            Decoder                ||
        # =======================================
        self.D0 = nn.Sequential(  # (64, 32, 32) -> (128, 64, 64)
            nn.Conv2d(64, 512, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2)
        )

        self.D1 = nn.Sequential(  # (384, 64, 64) -> (128, 128, 128)
            nn.Conv2d(128 + 256, 512, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2)
        )

        self.D2 = nn.Sequential(  # (256, 128, 128) -> (64, 256, 256)
            nn.Conv2d(128 + 128, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2)
        )

        self.D3 = nn.Sequential(  # (128, 256, 256) -> (32, 512, 512)
            nn.Conv2d(64 + 64, 128, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2)
        )

        self.D4 = nn.Sequential(  # (64, 512, 512) -> (16, 1024, 1024)
            nn.Conv2d(32 + 32, 64, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2)
        )

        self.D5 = nn.Conv2d(16 + 16, img_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, mapping, synthesis):
        # extract feature
        batch_size = x.shape[0]
        feat_E0 = self.E0(x)  # 32 x 32
        feat_E1 = self.E1(feat_E0)  # 16 x 16
        feat_E2 = self.E2(feat_E1)  # 8 x 8
        feat_E3 = self.E3(feat_E2)  # 4 x 4
        latent = self.E4_fc(self.E4_conv(feat_E3).view(batch_size, -1))

        # mapping
        latent_out = torch.nn.LeakyReLU(5)(mapping(latent))
        gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
        latent = torch.randn((batch_size, 18, 512), dtype=torch.float, requires_grad=False)
        latent_in = latent.expand(-1, 18, -1).to(self.device)
        dlatents_in = self.lrelu(latent_in * gaussian_fit["std"] + gaussian_fit["mean"])

        # noise
        noise_in = []
        for i in range(18):
            res = (batch_size, 1, 2 ** (i // 2 + 2), 2 ** (i // 2 + 2))
            new_noise = torch.randn(res, dtype=torch.float).to(self.device)
            new_noise.requires_grad = False
            noise_in.append(new_noise)

        x = None
        gs = []
        for i, m in enumerate(synthesis.blocks.values()):
            if i == 0:
                x = m(dlatents_in[:, 2 * i:2 * i + 2], noise_in[2 * i:2 * i + 2])
            else:
                if i == 1:
                    x = self.merge1(torch.cat([x, feat_E3], dim=1))  # 4
                elif i == 2:
                    x = self.merge2(torch.cat([x, feat_E2], dim=1))  # 8
                elif i == 3:
                    x = self.merge3(torch.cat([x, feat_E1], dim=1))  # 16
                elif i == 4:
                    x = self.merge4(torch.cat([x, feat_E0], dim=1))  # 32
                x = m(x, dlatents_in[:, 2 * i:2 * i + 2], noise_in[2 * i:2 * i + 2])

            gs.append(x)

        feat_D0 = self.D0(feat_E0)
        feat_D1 = self.D1(torch.cat([feat_D0, gs[4]], dim=1))
        feat_D2 = self.D2(torch.cat([feat_D1, gs[5]], dim=1))
        feat_D3 = self.D3(torch.cat([feat_D2, gs[6]], dim=1))
        feat_D4 = self.D4(torch.cat([feat_D3, gs[7]], dim=1))
        out = self.D5(torch.cat([feat_D4, gs[8]], dim=1))
        return out


if __name__ == '__main__':
    use_cuda = True
    device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
    synthesis = G_synthesis()
    mapping = G_mapping()
    print(device)
    model = GLEAN(device=device).to(device)
    input = torch.ones([2, 3, 32, 32]).to(device)
    out = model(input)
    print(out.shape)
