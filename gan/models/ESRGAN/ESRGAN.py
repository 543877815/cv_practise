import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from utils import get_platform_path
from .model import GeneratorRRDB, Discriminator, FeatureExtractor
import torch.backends.cudnn as cudnn
from gan.common import FloatTensor as Tensor
from gan.common import denormalize
from torchvision.utils import save_image, make_grid


class ESRGAN(object):
    def __init__(self, config, dataloader=None, device=None):
        super(ESRGAN, self).__init__()

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
        self.residual_blocks = config.residual_blocks
        self.filters = config.filters
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
        self.criterion_pixel = None
        self.warmup_batches = config.warmup_batches
        self.lambda_adv = float(config.lambda_adv)
        self.lambda_pixel = float(config.lambda_pixel)
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
        self.generator = GeneratorRRDB(channels=self.channels, filters=self.filters,
                                       num_res_blocks=self.residual_blocks)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.discriminator = Discriminator(input_shape=(self.channels, *self.hr_shape), )
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss()
        self.criterion_content = torch.nn.L1Loss()
        self.criterion_pixel = torch.nn.L1Loss()
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
            self.feature_extractor.cuda()

        if self.resume:
            self.generator.load_state_dict(torch.load(self.generator_checkpoint))
            self.discriminator.load_state_dict(torch.load(self.discriminator_checkpoint))

    def train(self):
        data_dir, _, checkpoint_dir, _ = get_platform_path()
        for epoch in range(self.start_epoch, self.epochs):
            for i, imgs in enumerate(self.dataloader):

                batches_done = epoch * len(self.dataloader) + i

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

                # Measure pixel-wise loss against ground truth
                loss_pixel = self.criterion_pixel(gen_hr, imgs_hr)

                if batches_done < self.warmup_batches:
                    # Warm-up (pixel-wise loss only)
                    loss_pixel.backward()
                    self.optimizer_G.step()
                    print(
                        "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                        % (epoch, self.epochs, i, len(self.dataloader), loss_pixel.item())
                    )
                    continue

                # Extract validity predictions from discriminator
                pred_real = self.discriminator(imgs_hr).detach()
                pred_fake = self.discriminator(gen_hr)

                # Adversarial loss (relativistic average GAN)
                loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

                # Content loss
                gen_features = self.feature_extractor(gen_hr)
                real_features = self.feature_extractor(imgs_hr).detach()
                loss_content = self.criterion_content(gen_features, real_features)

                # Total generator loss
                loss_G = loss_content + self.lambda_adv * loss_GAN + self.lambda_pixel * loss_pixel

                loss_G.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.optimizer_D.zero_grad()

                pred_real = self.discriminator(imgs_hr)
                pred_fake = self.discriminator(gen_hr.detach())

                # Adversarial loss for real and fake images (relativistic average GAN)
                loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
                loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

                # Total loss
                loss_D = (loss_real + loss_fake) / 2

                loss_D.backward()
                self.optimizer_D.step()

                # --------------
                #  Log Progress
                # --------------

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
                    % (
                        epoch,
                        self.epochs,
                        i,
                        len(self.dataloader),
                        loss_D.item(),
                        loss_G.item(),
                        loss_content.item(),
                        loss_GAN.item(),
                        loss_pixel.item(),
                    )
                )

                if batches_done % self.sample_interval == 0:
                    # Save image grid with upsampled inputs and SRGAN outputs
                    save_dir = '{}/{}'.format(data_dir, self.model_name)
                    if not os.path.exists(save_dir):
                        os.mkdir(save_dir)
                    imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=self.upscale_factor)
                    gen_hr = denormalize(make_grid(gen_hr[:4], nrow=1, normalize=True))
                    imgs_lr = make_grid(imgs_lr[:4], nrow=1, normalize=True)
                    imgs_hr = make_grid(imgs_hr[:4], nrow=1, normalize=True)
                    img_grid = torch.cat((imgs_lr, gen_hr, imgs_hr), -1)
                    save_image(img_grid, "{}/{}.png".format(save_dir, batches_done), normalize=False)

                if self.checkpoint_interval != -1 and epoch % self.checkpoint_interval == 0:
                    # Save model checkpoints
                    model_dir = '{}/{}'.format(checkpoint_dir, self.model_name)
                    if not os.path.exists(model_dir):
                        os.mkdir(model_dir)
                    torch.save(self.generator.state_dict(), "{}/generator_{}.pth".format(model_dir, epoch))
                    torch.save(self.discriminator.state_dict(), "{}/discriminator_{}.pth".format(model_dir, epoch))
