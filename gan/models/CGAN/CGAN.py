import torch
from torch.autograd import Variable
import numpy as np
from utils import get_platform_path
from .model import Generator, Discriminator
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import os
from gan.common import FloatTensor, LongTensor

class CGAN(object):
    def __init__(self, config, dataloader=None, device=None):
        super(CGAN, self).__init__()

        # hardware
        self.CUDA = torch.cuda.is_available()
        self.device = device

        # data configuration
        self.dataloader = dataloader

        # models configuration
        self.latent_dim = config.latent_dim
        self.img_size = (config.channels, config.img_size, config.img_size)
        self.n_classes = config.n_classes
        self.model_name = config.model

        # experiment configuration
        self.epochs = config.epoch
        self.lr = config.lr
        self.beta1 = config.beta1e
        self.beta2 = config.beta2
        self.generator = None
        self.optimizer_G = None
        self.discriminator = None
        self.optimizer_D = None
        self.criterion = None
        self.seed = 123

        # checkpoint
        self.sample_interval = config.sample_interval

        # build model
        self.build_model()

    def build_model(self):
        self.generator = Generator(n_classes=self.n_classes, latent_dim=self.latent_dim, img_size=self.img_size)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.discriminator = Discriminator(n_classes=self.n_classes, img_size=self.img_size)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.criterion = torch.nn.MSELoss()

        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.generator.cuda()
            self.discriminator.cuda()
            self.criterion.cuda()

    def train(self):
        # ----------
        #  Training
        # ----------
        for epoch in range(self.epochs):
            for i, (imgs, labels) in enumerate(self.dataloader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = Variable(labels.type(LongTensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                gen_labels = Variable(LongTensor(np.random.randint(0, self.n_classes, batch_size)))  # [batch, 10]

                # Generate a batch of images
                gen_imgs = self.generator(z, gen_labels)

                # Loss measures generator's ability to fool the discriminator
                validity = self.discriminator(gen_imgs, gen_labels)
                g_loss = self.criterion(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Loss for real images
                validity_real = self.discriminator(real_imgs, labels)
                d_real_loss = self.criterion(validity_real, valid)

                # Loss for fake images
                validity_fake = self.discriminator(gen_imgs.detach(), gen_labels)
                d_fake_loss = self.criterion(validity_fake, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.epochs, i, len(self.dataloader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.sample_interval == 0:
                    self.sample_image(n_row=10, batches_done=batches_done)

    def sample_image(self, n_row, batches_done):
        data_dir, _, _, _ = get_platform_path()
        FloatTensor = torch.cuda.FloatTensor if self.CUDA else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.CUDA else torch.LongTensor
        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Sample noise
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))))
        # Get labels ranging from 0 to n_classes for n rows
        labels = np.array([num for _ in range(n_row) for num in range(n_row)])
        labels = Variable(LongTensor(labels))
        gen_imgs = self.generator(z, labels)
        save_dir = '{}/{}'.format(data_dir, self.model_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        save_image(gen_imgs.data, "{}/{}.png".format(save_dir, batches_done),
                   nrow=n_row, normalize=True)