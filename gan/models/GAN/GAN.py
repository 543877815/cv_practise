import os
import torch
from torch.autograd import Variable
import numpy as np

from utils import get_platform_path
from .model import Generator, Discriminator
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
from gan.common import FloatTensor as Tensor


class GAN(object):
    def __init__(self, config, dataloader=None, device=None):
        super(GAN, self).__init__()

        # hardware
        self.CUDA = torch.cuda.is_available()
        self.device = device

        # data configuration
        self.dataloader = dataloader

        # models configuration
        self.latent_dim = config.latent_dim
        self.img_size = (config.channels, config.img_size, config.img_size)
        self.channels = config.channels
        self.model_name = config.model

        # experiment configuration
        self.epochs = config.epoch
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.generator = None
        self.optimizer_G = None
        self.discriminator = None
        self.optimizer_D = None
        self.criterion = None
        self.seed = config.seed

        # checkpoint
        self.sample_interval = config.sample_interval

        # build model
        self.build_model()

    def build_model(self):
        self.generator = Generator(latent_dim=self.latent_dim, img_size=self.img_size).to(self.device)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.discriminator = Discriminator(img_size=self.img_size).to(self.device)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.criterion = torch.nn.BCELoss()  # 交叉熵

        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.generator.cuda()
            self.discriminator.cuda()
            self.criterion.cuda()

    def train(self):
        data_dir, _, _, _ = get_platform_path()
        for epoch in range(self.epochs):
            for i, (imgs, _) in enumerate(self.dataloader):

                # Adversarial ground truths
                valid = Variable(Tensor(imgs.size(0), 1).fill_(1.0), requires_grad=False)  # [batch_size, 1]
                fake = Variable(Tensor(imgs.size(0), 1).fill_(0.0), requires_grad=False)  # [batch_size, 1]

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1,
                                                     (imgs.shape[0], self.latent_dim))))  # [batch_size, latend_dim]

                # Generate a batch of images
                gen_imgs = self.generator(z)

                # Loss measures generator's ability to fool the discriminator
                g_loss = self.criterion(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Measure discriminator's ability to classify real from generated samples
                real_loss = self.criterion(self.discriminator(real_imgs), valid)
                fake_loss = self.criterion(self.discriminator(gen_imgs.detach()), fake)  # 将gen_imgs从generator的计算图移除
                d_loss = (real_loss + fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.epochs, i, len(self.dataloader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.sample_interval == 0:

                    save_dir = '{}/{}'.format(data_dir, self.model_name)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    save_image(gen_imgs.data[:25], "{}/{}.png".format(save_dir, batches_done),
                               nrow=5, normalize=True)
