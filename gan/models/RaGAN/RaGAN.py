import torch
from torch.autograd import Variable
import numpy as np
import os
from utils import get_platform_path
from .model import Generator, Discriminator
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image


class RaGAN(object):
    def __init__(self, config, dataloader=None, device=None):
        super(RaGAN, self).__init__()
        self.CUDA = torch.cuda.is_available()
        if device is None:
            self.device = torch.device("cuda" if (config.use_cuda and self.CUDA) else "cpu")

        # data configuration
        self.dataloader = dataloader
        self.epochs = config.epoch
        self.lr = config.lr
        self.clip_value = config.clip_value
        self.n_critic = config.n_critic

        # models configuration
        self.latent_dim = config.latent_dim
        self.img_size = (config.channels, config.img_size, config.img_size)
        self.channels = config.channels
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.sample_interval = config.sample_interval
        self.rel_avg_gan = config.rel_avg_gan
        self.model_name = 'ragan'
        self.generator = None
        self.optimizer_G = None
        self.discriminator = None
        self.optimizer_D = None
        self.criterion = None

        self.seed = 123

    def build_model(self):
        self.generator = Generator(latent_dim=self.latent_dim, img_size=self.img_size).to(self.device)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.discriminator = Discriminator(img_size=self.img_size).to(self.device)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.criterion = torch.nn.BCEWithLogitsLoss().to(self.device)

        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True

    def train(self):
        self.build_model()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        data_dir, _, _, _ = get_platform_path()
        for epoch in range(self.epochs):
            for i, (imgs, _) in enumerate(self.dataloader):
                # Adversarial ground truths
                valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
                fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(Tensor))

                # ----------------
                # Train Generator
                # ----------------

                self.optimizer_G.zero_grad()

                # Sample noise as generator input
                z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], self.latent_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z)

                real_pred = self.discriminator(real_imgs).detach()
                fake_pred = self.discriminator(gen_imgs)

                if self.rel_avg_gan:
                    g_loss = self.criterion(fake_pred - real_pred.mean(0, keepdim=True), valid)
                else:
                    g_loss = self.criterion(fake_pred - real_pred, valid)

                # loss measures generator's ability to fool the discriminator
                g_loss = self.criterion(self.discriminator(gen_imgs), valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------
                # Train Discriminator
                # ---------------

                self.optimizer_D.zero_grad()

                # Predict validity
                real_pred = self.discriminator(real_imgs)
                fake_pred = self.discriminator(gen_imgs.detach())

                if self.rel_avg_gan:
                    real_loss = self.criterion(real_pred - fake_pred.mean(0, keepdim=True), valid)
                    fake_loss = self.criterion(fake_pred - real_pred.mean(0, keepdim=True), fake)
                else:
                    real_loss = self.criterion(real_pred - fake_pred, valid)
                    fake_loss = self.criterion(fake_pred - real_pred, fake)

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
