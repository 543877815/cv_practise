import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=200000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
args = parser.parse_args()
print(args)

img_shape = 100000

cuda = True if torch.cuda.is_available() else False


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(img_shape, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, img_shape),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(img_shape, 512),  # 连乘
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, img_shape),
            nn.Sigmoid(),
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()  # 交叉熵

# Initialize generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader
# os.makedirs("../../data/mnist", exist_ok=True)
# dataloader = torch.utils.data.DataLoader(
#     datasets.MNIST(
#         "../../data/mnist",
#         train=True,
#         download=True,
#         transform=transforms.Compose(
#             [transforms.Resize(args.img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
#         ),
#     ),
#     batch_size=args.batch_size,
#     shuffle=True,
# )

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

batch_size = 64

# Configure input
real_imgs = Variable(Tensor(np.random.normal(0, 1, (batch_size, img_shape))))

for epoch in range(args.epochs):

    # Adversarial ground truths
    valid = Variable(Tensor(batch_size, img_shape).fill_(1.0), requires_grad=False)  # size 0 is batch
    fake = Variable(Tensor(batch_size, img_shape).fill_(0.0), requires_grad=False)

    # -----------------
    #  Train Generator
    # -----------------

    optimizer_G.zero_grad()
    # Sample noise as generator input
    z = Variable(Tensor(np.random.uniform(-2, 2, (batch_size, img_shape))))
    # Generate a batch of images
    gen_imgs = generator(z)
    # Loss measures generator's ability to fool the discriminator
    g_loss = adversarial_loss(discriminator(gen_imgs), valid)

    g_loss.backward()
    optimizer_G.step()

    # ---------------------
    #  Train Discriminator
    # ---------------------

    optimizer_D.zero_grad()

    # Measure discriminator's ability to classify real from generated samples
    real_loss = adversarial_loss(discriminator(real_imgs), valid)
    fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2

    d_loss.backward()
    optimizer_D.step()

    print(
        "[Epoch %d/%d] [D loss: %f] [G loss: %f]"
        % (epoch, args.epochs, d_loss.item(), g_loss.item())
    )

    if epoch % args.sample_interval == 0:
        gen_imgs_cpu = gen_imgs[0].detach().cpu().numpy()
        real_imgs_cpu = real_imgs[0].detach().cpu().numpy()
        z_cpu = z[0].detach().cpu().numpy()
        fig = plt.figure(0)
        plt.hist(real_imgs_cpu, bins=100, alpha=0.5, facecolor='red', density=True, stacked=True)
        plt.hist(z_cpu, bins=100, alpha=0.5, facecolor='yellow', density=True, stacked=True)
        plt.hist(gen_imgs_cpu, bins=100, alpha=0.5, facecolor='blue', density=True, stacked=True)
        plt.savefig("images/%d.png" % epoch)
        plt.close(0)
