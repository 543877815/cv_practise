import torch
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
from gan.common import LongTensor, FloatTensor

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim, n_classes, code_dim, img_size, channels):
        super(Generator, self).__init__()
        input_dim = latent_dim + n_classes + code_dim                              # 62 + 10 + 2 = 74

        self.init_size = img_size // 4  # Initial size before upsampling           # 8
        self.l1 = nn.Sequential(nn.Linear(input_dim, 128 * self.init_size ** 2))   # [batch_size, 128 * 8 * 8]

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),                                                   # [batch_size, 128, 8, 8]
            nn.Upsample(scale_factor=2),                                           # [batch_size, 128, 16, 16]
            nn.Conv2d(128, 128, 3, stride=1, padding=1),                           # [batch_size, 128, 16, 16]
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),                                           # [batch_size, 128, 32, 32]
            nn.Conv2d(128, 64, 3, stride=1, padding=1),                            # [batch_size, 64, 32, 32]
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, channels, 3, stride=1, padding=1),                       # [batch_size, 1, 32, 32]
            nn.Tanh(),
        )

    def forward(self, noise, labels, code):
        gen_input = torch.cat((noise, labels, code), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self, channels, code_dim, n_classes, img_size):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            """Returns layers of each discriminator block"""
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(channels, 16, bn=False),                       # [batch_size, 16, 32, 32]
            *discriminator_block(16, 32),                                       # [batch_size, 32, 16, 16]
            *discriminator_block(32, 64),                                       # [batch_size, 64, 8, 8]
            *discriminator_block(64, 128),                                      # [batch_size, 128, 4, 4]
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4                                            # 32 / (2 ^ 4) = 2

        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))                        # [batch_size, 1]
        self.aux_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, n_classes), nn.Softmax())  # [batch_size, 10]
        self.latent_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, code_dim))              # [batch_size, 2]

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)
        latent_code = self.latent_layer(out)

        return validity, label, latent_code
