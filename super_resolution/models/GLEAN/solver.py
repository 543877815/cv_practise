# reference:
# https://github.com/icpm/super-resolution
# https://github.com/twtygqyy/pytorch-vdsr
from math import log10
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import os

from torch.autograd import Variable

from .GLEAN import GLEAN
from .Vgg19 import FeatureExtractor
from .RaGAN import Discriminator
from super_resolution.models.GLEAN.styleGAN import G_synthesis, G_mapping
from utils import progress_bar, get_platform_path, get_logger, shave, print_options
from torchvision.transforms import transforms
from PIL import Image
from torchvision import utils as vutils
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from collections import OrderedDict
from attrdict import AttrDict


class GLEANBasic(object):
    def __init__(self, config, device=None):
        super(GLEANBasic, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = device

        # models configuration
        self.generator = None
        self.img_channels = config.img_channels
        self.img_size = config.img_size
        self.upscale_factor = config.upscaleFactor
        self.test_upscaleFactor = config.test_upscaleFactor
        self.model_name = "{}-{}x".format(config.model, self.upscale_factor)

        # checkpoint configuration
        self.resume = config.resume
        self.checkpoint_name = "{}.pth".format(self.model_name)
        self.best_quality = 0
        self.start_epoch = 1
        self.n_epochs = config.n_epochs
        self.checkpoint_interval = config.checkpoint_interval

        # logger configuration
        _, _, _, log_dir = get_platform_path()
        self.logger = get_logger("{}/{}.log".format(log_dir, self.model_name))
        self.logger.info(print_options(config))

        # tensorboard writer
        self.writer = SummaryWriter(config.tensorboard_log_dir)
        self.tensorboard_image_interval = config.tensorboard_image_interval
        self.tensorboard_draw_model = config.tensorboard_draw_model
        self.tensorboard_input = config.tensorboard_input
        self.tensorboard_image_size = config.tensorboard_image_size
        self.tensorboard_image_sample = config.tensorboard_image_sample

        # memory bank
        self.synthesis = G_synthesis()
        self.mapping = G_mapping()

        # distributed
        self.distributed = config.distributed
        self.local_rank = config.local_rank

    def load_model(self):
        _, _, checkpoint_dir, _ = get_platform_path()
        print('==> Resuming from checkpoint...')
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('{}/{}'.format(checkpoint_dir, self.checkpoint_name))
        new_state_dict = OrderedDict()
        for key, value in checkpoint['net'].items():
            key = key.replace('module.', '')
            new_state_dict[key] = value
        self.generator.load_state_dict(new_state_dict)
        self.best_quality = checkpoint['psnr']
        self.start_epoch = checkpoint['epoch'] + 1
        self.logger.info("Start from epoch {}, best PSNR: {}".format(self.start_epoch, self.best_quality))

    # Deprecated (Too ugly...)
    def convert_BICUBIC(self, img):
        img_BICUBIC = torch.empty(img.shape[0], img.shape[1], img.shape[2] * self.test_upscaleFactor,
                                  img.shape[3] * self.test_upscaleFactor)
        for i in range(len(img)):
            x, y = img[i].shape[1:]
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((x * self.test_upscaleFactor, y * self.test_upscaleFactor),
                                  interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ])
            img_BICUBIC[i] = transform(img[i])
        return img_BICUBIC

    @staticmethod
    def psrn(mse):
        return 10 * log10(1 / mse)


class GLEANTrainer(GLEANBasic):
    def __init__(self, config, train_loader=None, test_loader=None, device=None):
        super(GLEANTrainer, self).__init__(config, device)

        self.seed = config.seed

        # parameters configuration
        self.lr = config.lr
        self.milestones = config.milestones
        self.scheduler_gamma = config.scheduler_gamma
        self.warmup_batches = config.warmup_batches
        self.beta1 = config.beta1
        self.beta2 = config.beta2

        # trade-off
        self.lambda_adv = float(config.lambda_adv)
        self.lambda_pixel = float(config.lambda_pixel)

        self.scheduler = None

        # model
        self.generator = None
        self.discriminator = None
        self.feature_extractor = None

        # optimizer
        self.optimizer_G = None
        self.optimizer_D = None

        # criterion
        self.criterion_pixel = None
        self.criterion_GAN = None
        self.criterion_content = None

        # data loader
        self.train_loader = train_loader
        self.test_loader = test_loader

        # pretrain_dir
        model_flist = AttrDict(config.model_flist[config.platform])
        self.synthesis_dir = model_flist.synthesis_dir
        self.mapping_dir = model_flist.mapping_dir

        # models init
        self.build_model()

    def build_model(self):
        # initialize model
        self.generator = GLEAN(img_channels=self.img_channels).to(self.device)
        self.discriminator = Discriminator(input_shape=(self.img_channels, self.img_size, self.img_size)).to(
            self.device)
        self.feature_extractor = FeatureExtractor().to(self.device)

        # resume
        if self.resume:
            self.load_model()

        # load pretrained GAN
        self.synthesis.load_state_dict(torch.load(self.synthesis_dir))
        self.mapping.load_state_dict(torch.load(self.mapping_dir))
        # set criterion
        self.criterion_pixel = torch.nn.L1Loss()
        self.criterion_GAN = torch.nn.BCEWithLogitsLoss()
        self.criterion_content = torch.nn.L1Loss()

        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion_pixel.cuda()
            self.criterion_GAN.cuda()
            self.criterion_content.cuda()
            self.synthesis.cuda()
            self.mapping.cuda()

        self.synthesis.eval()
        self.mapping.eval()
        self.feature_extractor.eval()

        # set optimizer and scheduler
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer_G,
                                                              milestones=self.milestones,
                                                              gamma=self.scheduler_gamma)

    def save_model(self, epoch):
        _, _, checkpoint_dir, _ = get_platform_path()
        name = self.checkpoint_name.replace('.pth', '_{}.pth'.format(epoch))
        g_model_out_path = '{}/G_{}'.format(checkpoint_dir, name)
        d_model_out_path = '{}/D_{}'.format(checkpoint_dir, name)
        torch.save(self.generator.state_dict(), g_model_out_path)
        torch.save(self.discriminator.state_dict(), d_model_out_path)
        self.logger.info("checkpoint saved to {} and {}".format(g_model_out_path, d_model_out_path))

    def train(self, epoch):
        self.generator.train()
        Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
        train_loss = 0
        for index, (lr, hr) in enumerate(self.train_loader):
            lr, hr = lr.to(self.device), hr.to(self.device)
            batches_done = (epoch - 1) * len(self.train_loader) + index

            # Adversarial ground truths
            valid = Variable(Tensor(np.ones((lr.size(0), *self.discriminator.output_shape))), requires_grad=False)
            fake = Variable(Tensor(np.zeros((lr.size(0), *self.discriminator.output_shape))), requires_grad=False)

            # ------------------
            #  Train Generators
            # ------------------
            self.optimizer_G.zero_grad()

            # Generate a high resolution image from low resolution input
            gen_hr = self.generator(lr, self.mapping, self.synthesis)

            # Measure pixel-wise loss against ground truth
            loss_pixel = self.criterion_pixel(gen_hr, hr)

            if batches_done < self.warmup_batches:
                # Warm-up (pixel-wise loss only)
                loss_pixel.backward()
                self.optimizer_G.step()
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
                    % (epoch, self.n_epochs, index, len(self.train_loader), loss_pixel.item())
                )
                continue

            # Extract validity predictions from discriminator
            pred_real = self.discriminator(hr).detach()
            pred_fake = self.discriminator(gen_hr)

            # Adversarial loss (relativistic average GAN)
            loss_GAN = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), valid)

            # Content loss
            gen_features = self.feature_extractor(gen_hr)
            real_features = self.feature_extractor(hr).detach()
            loss_content = self.criterion_content(gen_features, real_features)

            # Total generator loss
            loss_G = loss_content + self.lambda_adv * loss_GAN + self.lambda_pixel * loss_pixel
            loss_G.backward()
            self.optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.optimizer_D.zero_grad()

            pred_real = self.discriminator(hr)
            pred_fake = self.discriminator(gen_hr.detach())

            # Adversarial loss for real and fake images (relativistic average GAN)
            loss_real = self.criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
            loss_fake = self.criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)

            # Total loss
            loss_D = (loss_real + loss_fake) / 2
            self.optimizer_D.step()

            train_loss += loss_pixel.item()
            if not self.distributed or self.local_rank == 0:
                progress_bar(index, len(self.train_loader),
                             "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, content: %f, adv: %f, pixel: %f]"
                             % (
                                 epoch,
                                 self.n_epochs,
                                 index,
                                 len(self.train_loader),
                                 loss_D.item(),
                                 loss_G.item(),
                                 loss_content.item(),
                                 loss_GAN.item(),
                                 loss_pixel.item(),
                             ))

        avg_train_loss = train_loss / len(self.train_loader)
        print("    Average Loss: {:.4f}".format(avg_train_loss))
        return avg_train_loss

    def test(self):
        self.generator.eval()
        psnr = 0
        # random sample output to tensorboard
        save_inputs, save_outputs, save_targets = [], [], []
        with torch.no_grad():
            for index, (img, target) in enumerate(self.test_loader):
                img, target = img.to(self.device), target.to(self.device)
                output = self.generator(img).clamp(0.0, 1.0)
                output, target = shave(output, target, self.test_upscaleFactor)
                loss = ((output - target) ** 2).mean()
                psnr += self.psrn(loss.item())
                if not self.distributed or self.local_rank == 0:
                    progress_bar(index, len(self.test_loader), 'PSNR: %.4f' % (psnr / (index + 1)))
                    if index < self.tensorboard_image_sample:
                        save_inputs.append(img)
                        save_outputs.append(output)
                        save_targets.append(target)

        avg_psnr = psnr / len(self.test_loader)
        print("    Average PSNR: {:.4f} dB".format(avg_psnr))
        return avg_psnr, save_inputs, save_outputs, save_targets

    def run(self):
        for epoch in range(self.start_epoch, self.n_epochs + self.start_epoch):
            print('\n===> Epoch {} starts:'.format(epoch))
            avg_train_loss = self.train(epoch)
            avg_psnr, save_input, save_output, save_target = self.test()
            self.scheduler.step(epoch)
            if not self.distributed or self.local_rank == 0:

                # save to logger
                self.logger.info(
                    "Epoch [{}/{}]: lr={:.6f} loss={:.6f} PSNR={:.6f}".format(epoch,
                                                                              self.n_epochs + self.start_epoch,
                                                                              self.scheduler.get_lr()[0],
                                                                              avg_train_loss,
                                                                              avg_psnr))

                # save interval models
                if epoch % self.checkpoint_interval == 0:
                    self.save_model(epoch)

                # tensorboard scalars
                self.writer.add_scalar('train_loss', avg_train_loss, epoch)
                self.writer.add_scalar('psnr', avg_psnr, epoch)

                # tensorboard graph
                if epoch == 0 and self.tensorboard_draw_model and \
                        len(save_input) > 0 and len(save_target) > 0:
                    self.writer.add_graph(model=self.generator, input_to_model=[save_input[0]])

                # tensorboard images
                if epoch % self.tensorboard_image_interval == 0:

                    assert len(save_input) == len(save_output) == len(save_target), \
                        'the size of save_input and save_output and save_target is not equal.'
                    for i in range(len(save_target)):
                        save_input[i] = F.interpolate(save_input[i], size=self.tensorboard_image_size,
                                                      mode='bicubic', align_corners=True)
                        save_output[i] = F.interpolate(save_output[i], size=self.tensorboard_image_size,
                                                       mode='bicubic', align_corners=True)
                        save_target[i] = F.interpolate(save_target[i], size=self.tensorboard_image_size,
                                                       mode='bicubic', align_corners=True)
                        images = torch.cat((save_input[i], save_output[i], save_target[i]))
                        grid = vutils.make_grid(images)
                        self.writer.add_image('image-{}'.format(i), grid, epoch)
