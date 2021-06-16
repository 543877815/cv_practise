import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from utils import get_platform_path
from .model import GeneratorResNet, Discriminator, FeatureExtractor
import torch.backends.cudnn as cudnn
from gan.common import FloatTensor as Tensor
from torchvision.utils import save_image, make_grid


class SRGAN(object):
    def __init__(self, config, dataloader=None, device=None):
        super(SRGAN, self).__init__()

        # hardware
        self.CUDA = torch.cuda.is_available()
        self.device = device

        # data configuration
        self.dataloader = dataloader

        # models configuration
        self.latent_dim = config.latent_dim
        self.img_size = config.img_size
        self.channels = config.channels
        self.model_name = config.model
        self.hr_shape = (config.hr_height, config.hr_width)
        self.n_residual_blocks = config.n_residual_blocks
        self.upscale_factor = config.upscale_factor

        # experiment configuration
        self.epochs = config.epochs
        self.start_epoch = config.start_epoch
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.generator = None
        self.optimizer_G = None
        self.discriminator = None
        self.optimizer_D = None
        self.feature_extractor = None
        self.criterion_GAN = None
        self.criterion_content = None
        self.seed = config.seed

        # checkpoint
        self.sample_interval = config.sample_interval
        self.checkpoint_interval = config.checkpoint_interval
        self.resume = config.resume
        self.generator_checkpoint = config.generator_checkpoint
        self.discriminator_checkpoint = config.discriminator_checkpoint

        # build model
        self.build_model()

    def build_model(self):
        self.generator = GeneratorResNet(in_channels=self.channels, out_channels=self.channels,
                                         n_residual_blocks=self.n_residual_blocks)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.discriminator = Discriminator(input_shape=(self.channels, *self.hr_shape), )
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.criterion_GAN = torch.nn.MSELoss()  # Reconstruction loss of AE
        self.criterion_content = torch.nn.L1Loss()  # Reconstruction loss of AE
        self.feature_extractor = FeatureExtractor()

        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.generator.cuda()
            self.discriminator.cuda()
            self.feature_extractor.cuda()
            self.criterion_GAN.cuda()
            self.criterion_content.cuda()

        if self.resume:
            self.generator.load_state_dict(torch.load(self.generator_checkpoint))
            self.discriminator.load_state_dict(torch.load(self.discriminator_checkpoint))

    def pullaway_loss(self, embeddings):
        norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
        normalized_emb = embeddings / norm
        similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
        batch_size = embeddings.size(0)
        loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
        return loss_pt

    def train(self):
        data_dir, _, checkpoint_dir, _ = get_platform_path()
        for epoch in range(self.start_epoch, self.epochs):
            for i, imgs in enumerate(self.dataloader):

                # Configure model input
                imgs_lr = Variable(imgs["lr"].type(Tensor))
                imgs_hr = Variable(imgs["hr"].type(Tensor))

                # Adversarial ground truths
                valid = Variable(Tensor(np.ones((imgs_lr.size(0), *self.discriminator.output_shape))),
                                 requires_grad=False)
                fake = Variable(Tensor(np.zeros((imgs_lr.size(0), *self.discriminator.output_shape))),
                                requires_grad=False)

                # ------------------
                #  Train Generators
                # ------------------

                self.optimizer_G.zero_grad()

                # Generate a high resolution image from low resolution input
                gen_hr = self.generator(imgs_lr)

                # Adversarial loss
                loss_GAN = self.criterion_GAN(self.discriminator(gen_hr), valid)

                # Content loss
                gen_features = self.feature_extractor(gen_hr)
                real_features = self.feature_extractor(imgs_hr)
                loss_content = self.criterion_content(gen_features, real_features.detach())

                # Total loss
                loss_G = loss_content + 1e-3 * loss_GAN

                loss_G.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Loss of real and fake images
                loss_real = self.criterion_GAN(self.discriminator(imgs_hr), valid)
                loss_fake = self.criterion_GAN(self.discriminator(gen_hr.detach()), fake)

                # Total loss
                loss_D = (loss_real + loss_fake) / 2

                loss_D.backward()
                self.optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n"
                    % (epoch, self.epochs, i, len(self.dataloader), loss_D.item(), loss_G.item())
                )

                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.sample_interval == 0:
                    # Save image grid with upsampled inputs and SRGAN outputs
                    save_dir = '{}/{}'.format(data_dir, self.model_name)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=self.upscale_factor)
                    gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
                    imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
                    img_grid = torch.cat((imgs_lr, gen_hr), -1)
                    save_image(img_grid, "{}/{}.png".format(save_dir, batches_done), normalize=False)

            if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
                # Save model checkpoints
                model_dir = '{}/{}'.format(checkpoint_dir, self.model_name)
                if not os.path.exists(model_dir):
                    os.mkdir(model_dir)
                torch.save(self.generator.state_dict(), "{}/generator_{}.pth".format(model_dir, epoch))
                torch.save(self.discriminator.state_dict(), "{}/discriminator_{}.pth".format(model_dir, epoch))
