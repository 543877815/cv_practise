import os
import torch
from torch.autograd import Variable
import numpy as np

from utils import get_platform_path
from .model import Generator, Discriminator
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image


class WGAN(object):
    def __init__(self, config, dataloader=None, device=None):
        super(WGAN, self).__init__()

        # hardware
        self.CUDA = torch.cuda.is_available()
        self.device = device

        # data configuration
        self.dataloader = dataloader

        # models configuration
        self.latent_dim = config.latent_dim
        self.img_size = (config.channels, config.img_size, config.img_size)
        self.model_name = config.model
        self.channels = config.channels

        # experiment configuration
        self.epochs = config.epoch
        self.lr = config.lr
        self.clip_value = config.clip_value
        self.n_critic = config.n_critic
        self.sample_interval = config.sample_interval
        self.generator = None
        self.optimizer_G = None
        self.discriminator = None
        self.optimizer_D = None
        self.seed = 123

        # checkpoint
        self.sample_interval = config.sample_interval

        # build model
        self.build_model()

    def build_model(self):
        self.generator = Generator(latent_dim=self.latent_dim, img_size=self.img_size).to(self.device)
        self.optimizer_G = torch.optim.RMSprop(self.generator.parameters(), lr=self.lr)
        self.discriminator = Discriminator(latent_dim=self.latent_dim, img_size=self.img_size).to(self.device)
        self.optimizer_D = torch.optim.RMSprop(self.discriminator.parameters(), lr=self.lr)

        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True

    def train(self):
        self.build_model()
        data_dir, _, _, _ = get_platform_path()
        Tensor = torch.cuda.FloatTensor if self.CUDA else torch.FloatTensor
        batches_done = 0
        for epoch in range(self.epochs):
            for i, (imgs, _) in enumerate(self.dataloader):

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                fake_imgs = self.generator(z).detach()
                # Adversarial loss
                loss_D = -torch.mean(self.discriminator(real_imgs)) + torch.mean(self.discriminator(fake_imgs))

                loss_D.backward()
                self.optimizer_D.step()

                # Clip weights of discriminator
                for p in self.discriminator.parameters():
                    p.data.clamp_(-self.clip_value, self.clip_value)

                # Train the generator every n_critic iterations
                if i % self.n_critic == 0:
                    # -----------------
                    #  Train Generator
                    # -----------------

                    self.optimizer_G.zero_grad()

                    # Generate a batch of images
                    gen_imgs = self.generator(z)
                    # Adversarial loss
                    loss_G = -torch.mean(self.discriminator(gen_imgs))

                    loss_G.backward()
                    self.optimizer_G.step()

                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                        % (epoch, self.epochs, batches_done % len(self.dataloader), len(self.dataloader), loss_D.item(),
                           loss_G.item())
                    )

                if batches_done % self.sample_interval == 0:
                    save_dir = '{}/{}'.format(data_dir, self.model_name)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    save_image(gen_imgs.data[:25], "{}/{}.png".format(save_dir, batches_done),
                               nrow=5, normalize=True)
                batches_done += 1
