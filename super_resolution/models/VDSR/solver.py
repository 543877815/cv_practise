# reference:
# https://github.com/icpm/super-resolution
# https://github.com/twtygqyy/pytorch-vdsr
from math import log10

import torch
import torch.backends.cudnn as cudnn
import os

from torch.autograd import Variable

from .model import VDSR
from utils import progress_bar, get_platform_path, get_logger
from torchvision.transforms import transforms
from PIL import Image
from torchvision import utils as vutils
import torch.nn as nn


class VSDRBasic(object):
    def __init__(self, config, device=None):
        super(VSDRBasic, self).__init__()
        self.CUDA = torch.cuda.is_available()
        if device is None:
            self.device = torch.device("cuda" if (config.use_cuda and self.CUDA) else "cpu")

        # model configuration
        self.model = None
        self.color_space = config.color_space
        self.filter = config.filter
        self.num_residuals = config.num_residuals
        self.num_channels = config.num_channels
        self.upscale_factor = config.upscaleFactor
        self.test_upscaleFactor = config.test_upscaleFactor
        self.model_name = "{}-{}x".format(config.model, self.upscale_factor)

        # checkpoint configuration
        self.resume = config.resume
        self.checkpoint_name = "{}.pth".format(self.model_name)
        self.best_quality = 0
        self.start_epoch = 1

        # parameters
        self.momentum = config.momentum
        self.scheduler_gamma = config.scheduler_gamma
        self.weight_decay = config.weight_decay
        self.milestones = config.milestones

        # logger configuration
        _, _, _, log_dir = get_platform_path()
        self.logger = get_logger("{}/{}.log".format(log_dir, self.model_name))
        self.logger.info(config)

    def load_model(self):
        _, _, checkpoint_dir, _ = get_platform_path()
        print('==> Resuming from checkpoint...')
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('{}/{}'.format(checkpoint_dir, self.checkpoint_name))
        self.model.load_state_dict(checkpoint['net'])
        self.best_quality = checkpoint['psnr']
        self.start_epoch = checkpoint['epoch'] + 1

    def convert_BICUBIC(self, img):
        img_BICUBIC = torch.empty(img.shape[0], img.shape[1], img.shape[2] * self.test_upscaleFactor,
                                  img.shape[3] * self.test_upscaleFactor)
        for i in range(len(img)):
            x, y = img[i].shape[1:]
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((x * self.test_upscaleFactor, y * self.test_upscaleFactor), interpolation=Image.BICUBIC),
                transforms.ToTensor(),
            ])
            img_BICUBIC[i] = transform(img[i])
        return img_BICUBIC

    def convert_same(self, img, target):
        target_new = torch.empty((img.shape))

        for i in range(len(img)):
            x, y = img[i].shape[1:]
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((x, y), interpolation=Image.BICUBIC),
                transforms.ToTensor()
            ])
            target_new[i] = transform(target[i])
        return target_new

    @staticmethod
    def psrn(mse):
        return 10 * log10(1 / mse)


class VDSRTester(VSDRBasic):
    def __init__(self, config, test_loader=None, device=None):
        super(VDSRTester, self).__init__(config)
        assert (config.resume is True)

        data_dir, _, _, _ = get_platform_path()
        # resolve configuration
        self.output = data_dir + config.output
        self.test_loader = test_loader
        self.criterion = torch.nn.MSELoss(reduction='sum')

    def build_model(self):
        self.model = VDSR(num_channels=self.num_channels, filter=self.filter, num_residuals=self.num_residuals).to(
            self.device)
        self.load_model()
        if self.CUDA:
            cudnn.benchmark = True
            self.criterion.cuda()

    def run(self):
        self.build_model()
        self.model.eval()
        with torch.no_grad():
            for index, (img, filename) in enumerate(self.test_loader):
                img_BICUBIC = self.convert_BICUBIC(img)
                img_BICUBIC = img_BICUBIC.to(self.device)
                # full RGB/YCrCb
                if self.num_channels == 3:
                    output = self.model(img_BICUBIC).clamp(0.0, 1.0).cpu()
                # y
                elif self.num_channels == 1:
                    output = self.model(img_BICUBIC[:, 0, :, :].unsqueeze(1))
                    img_BICUBIC[:, 0, :, :].data = output
                    output = img_BICUBIC.clamp(0.0, 1.0).cpu()
                assert (len(output.shape) == 4 and output.shape[0] == 1)
                output_name = filename[0].replace('LR', 'HR')
                if self.color_space == 'RGB':
                    vutils.save_image(output, self.output + output_name)
                elif self.color_space == 'YCbCr':
                    output = transforms.ToPILImage(mode='YCbCr')(output[0]).convert("RGB")
                    output.save(self.output + output_name)
                print('==> {} is saved to {}'.format(output_name, self.output))


class VDSRTrainer(VSDRBasic):
    def __init__(self, config, train_loader=None, test_loader=None, device=None, clip=0.4):
        super(VDSRTrainer, self).__init__(config, device)

        # model configuration
        self.lr = config.lr

        # checkpoint configuration
        self.epochs = config.epochs

        # parameters configuration
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.clip = clip

    def build_model(self):
        self.model = VDSR(num_channels=self.num_channels, filter=self.filter, num_residuals=self.num_residuals).to(
            self.device)

        if self.resume:
            self.load_model()
        # else:
        #     self.model.weight_init()

        self.criterion = torch.nn.MSELoss(reduction='sum')
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.weight_decay,
                                         weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones,
                                                              gamma=self.scheduler_gamma)

    def save_model(self, epoch, avg_psnr):
        _, _, checkpoint_dir, _ = get_platform_path()
        model_out_path = '{}/{}'.format(checkpoint_dir, self.checkpoint_name)
        state = {
            'net': self.model.state_dict(),
            'psnr': avg_psnr,
            'epoch': epoch
        }
        torch.save(state, model_out_path)
        self.logger.info("checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0

        for index, (img, target) in enumerate(self.train_loader):
            if img.shape != target.shape:
                img_BICUBIC = self.convert_BICUBIC(img)
            else:
                img_BICUBIC = img

            img_BICUBIC, target = img_BICUBIC.to(self.device), target.to(self.device)

            output = self.model(img_BICUBIC)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()

            train_loss += loss.item()
            progress_bar(index, len(self.train_loader), 'Loss: %.4f' % (train_loss / (index + 1)))

        avg_train_loss = train_loss / len(self.train_loader)
        print("    Average Loss: {:.4f}".format(avg_train_loss))

        return avg_train_loss

    def test(self):
        self.model.eval()
        psnr = 0
        with torch.no_grad():
            for index, (img, target) in enumerate(self.test_loader):
                if img.shape != target.shape:
                    img_BICUBIC = self.convert_BICUBIC(img)
                    target = self.convert_same(img_BICUBIC, target)
                else:
                    img_BICUBIC = img
                img_BICUBIC, target = img_BICUBIC.to(self.device), target.to(self.device)

                output = self.model(img_BICUBIC).clamp(0.0, 1.0)
                loss = self.criterion(output, target)

                psnr += self.psrn(loss.item() / target.shape[2] / target.shape[3])
                progress_bar(index, len(self.test_loader), 'PSNR: %.4f' % (psnr / (index + 1)))

        avg_psnr = psnr / len(self.test_loader)
        print("    Average PSNR: {:.4f} dB".format(avg_psnr))
        return avg_psnr

    def run(self):
        self.build_model()
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            print('\n===> Epoch {} starts:'.format(epoch))
            avg_train_loss = self.train()
            avg_psnr = self.test()
            self.scheduler.step()
            self.logger.info("Epoch [{}/{}]: loss={} PSNR={}".format(epoch, self.epochs + self.start_epoch,
                                                                     avg_train_loss, avg_psnr))

            if avg_psnr > self.best_quality:
                self.best_quality = avg_psnr
                self.save_model(epoch, avg_psnr)
