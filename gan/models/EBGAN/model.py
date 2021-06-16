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
    def __init__(self, img_size, latent_dim, channels):
        super(Generator, self).__init__()

        self.init_size = img_size // 4  # 8
        self.l1 = nn.Sequential(nn.Linear(latent_dim, 128 * self.init_size ** 2)) # [batch_size, 128 * 16]

        self.conv_blocks = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)   # [batch_size, 128, 8, 8]
        img = self.conv_blocks(out)                                         # [batch_size, 128, 32, 32]
        return img


class Discriminator(nn.Module):
    def __init__(self, channels, img_size):
        super(Discriminator, self).__init__()

        # Upsampling
        self.down = nn.Sequential(nn.Conv2d(channels, 64, 3, 2, 1), nn.ReLU())  # [batch_size, 64, 16, 16] downsampling
        # Fully-connected layers
        self.down_size = img_size // 2          # 16
        down_dim = 64 * (img_size // 2) ** 2    # 64 * 16 * 16

        self.embedding = nn.Linear(down_dim, 32)    # [batch_size, 32]

        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        # Upsampling
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, channels, 3, 1, 1))

    def forward(self, img):
        out = self.down(img)                                     # [batch_size, 64, 16, 16]
        embedding = self.embedding(out.view(out.size(0), -1))    # [batch_size, 64 * 16 * 16] -> [batch_size, 32]
        out = self.fc(embedding)                                 # [batch_size, 32] -> [batch_size, 64 * 16 * 16]
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size))  # [batch_size, 64, 16, 16]
        return out, embedding