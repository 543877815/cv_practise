import torch.nn as nn
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim, img_size, channels):
        super(Generator, self).__init__()

        self.init_size = img_size // 4                   # 32 // 4 = 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2))  # [100, 128 * 16]

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),                                            # [batch_size, 128, 8, 8] 2D里面做归一化
            nn.Upsample(scale_factor=2),                                    # [batch_size, 128, 16, 16]
            nn.Conv2d(128, 128, 3, stride=1, padding=1),                    # [batch_size, 128, 16, 16]
            nn.BatchNorm2d(128, 0.8),                                       # [batch_size, 128, 16, 16]
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),                                    # [batch_size, 128, 32, 32]
            nn.Conv2d(128, 64, 3, stride=1, padding=1),                     # [batch_size, 128, 32, 32]
            nn.BatchNorm2d(64, 0.8),                                        # 0.8 is momentum
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)   # [batch_size, 128, 8, 8]
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels, img_size):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1), nn.Sigmoid())

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity
