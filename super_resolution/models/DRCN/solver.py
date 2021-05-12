# reference:
# https://github.com/icpm/super-resolution
# https://github.com/NJU-Jet/SR_Framework
# https://github.com/jiny2001/deeply-recursive-cnn-tf/blob/master/super_resolution.py

from math import log10

import torch
import torch.backends.cudnn as cudnn
import os

from .model import DRCN
from utils import progress_bar, get_platform_path, get_logger
from torchvision.transforms import transforms
from PIL import Image
from torchvision import utils as vutils


class DRCNBasic(object):
    def __init__(self, config, device=None):
        super(DRCNBasic, self).__init__()
        self.CUDA = torch.cuda.is_available()
        if device is None:
            self.device = torch.device("cuda" if (config.use_cuda and self.CUDA) else "cpu")

        # model configuration
        self.model = None
        self.color_space = config.color
        self.num_channels = config.num_channels
        self.upscale_factor = config.upscaleFactor
        self.num_recursions = 16
        self.model_name = "DRCN-{}x".format(self.upscale_factor)

        # checkpoint configuration
        self.resume = config.resume
        self.checkpoint_name = "{}.pth".format(self.model_name)
        self.best_quality = 0
        self.start_epoch = 1

        # logger configuration
        _, _, _, log_dir = get_platform_path()
        self.logger = get_logger("{}/{}.log".format(log_dir, self.model_name))
        self.logger.info(config)

        self.metric = 0  # used for learning rate policy 'plateau'

    def load_model(self):
        _, _, checkpoint_dir, _ = get_platform_path()
        print('==> Resuming from checkpoint...')
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('{}/{}'.format(checkpoint_dir, self.checkpoint_name))
        self.model.load_state_dict(checkpoint['net'])
        self.best_quality = checkpoint['psnr']
        self.start_epoch = checkpoint['epoch'] + 1
        print("best quality: {}".format(self.best_quality))

    def convert_BICUBIC(self, img):
        img_BICUBIC = torch.empty(img.shape[0], img.shape[1], img.shape[2] * self.upscale_factor,
                                  img.shape[3] * self.upscale_factor)
        for i in range(len(img)):
            x, y = img[i].shape[1:]
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((x * self.upscale_factor, y * self.upscale_factor), interpolation=Image.BICUBIC),
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


class DRCNTester(DRCNBasic):
    def __init__(self, config, test_loader=None, device=None):
        super(DRCNTester, self).__init__(config, device)
        assert (config.resume is True)

        data_dir, _, _, _ = get_platform_path()
        # resolve configuration
        self.output = data_dir + config.output
        self.test_loader = test_loader
        self.criterion = torch.nn.MSELoss(reduction='mean')

    def build_model(self):
        self.model = DRCN(num_channels=self.num_channels, num_recursions=self.num_recursions).to(
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


class DRCNTrainer(DRCNBasic):
    def __init__(self, config, train_loader=None, test_loader=None, device=None, clip=0.01):
        super(DRCNTrainer, self).__init__(config, device)

        # model configuration

        self.loss_alpha = 1.0
        self.clip = clip
        self.lr = config.lr
        # self.loss_beta, self.num_recursions = 0.001, 1
        # self.loss_beta, self.num_recursions = 0.0000005, 9
        self.loss_beta, self.num_recursions = 0.00001, 16
        self.loss_alpha_zero_epoch = 25
        self.loss_alpha_decay = self.loss_alpha / self.loss_alpha_zero_epoch

        # checkpoint configuration
        self.epochs = config.epochs

        # parameters configuration
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed

        self.train_loader = train_loader
        self.test_loader = test_loader

    def build_model(self):
        self.model = DRCN(num_channels=self.num_channels, num_recursions=self.num_recursions).to(
            self.device)

        if self.resume:
            self.load_model()

        self.criterion = torch.nn.MSELoss(reduction='mean')
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=25,
                                                                    min_lr=1e-6, threshold=1e-4, verbose=True)

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
        train_reg = 0
        train_loss2 = 0
        train_loss1 = 0
        train_loss = 0
        for index, (img, target) in enumerate(self.train_loader):
            if img.shape != target.shape:
                img_BICUBIC = self.convert_BICUBIC(img)
            else:
                img_BICUBIC = img

            img_BICUBIC, target = img_BICUBIC.to(self.device), target.to(self.device)

            target_d, output = self.model(img_BICUBIC)

            loss1 = 0
            for i in range(self.num_recursions + 1):
                loss1 += self.criterion(target_d[:, i, :, :].unsqueeze(1), target) / (self.num_recursions + 1)

            loss2 = self.criterion(output, target)

            # regularization
            reg_term = 0
            for theta in self.model.parameters():
                reg_term += torch.mean(torch.sum(theta ** 2))

            loss = self.loss_alpha * loss1 + (1 - self.loss_alpha) * loss2 + reg_term * self.loss_beta
            self.optimizer.zero_grad()

            loss.backward()

            self.optimizer.step()

            train_reg += self.loss_beta * reg_term.item()
            train_loss2 += loss2.item()
            train_loss1 += loss1.item()
            train_loss += loss.item()

            progress_bar(index, len(self.train_loader), 'Loss: %.4f, Loss1: %4f, Loss2: %4f, reg_term: %4f' % (
                train_loss / (index + 1), train_loss1 / (index + 1),
                train_loss2 / (index + 1), train_reg / (index + 1)))

        avg_train_loss = train_loss / len(self.train_loader)
        print("    Average Loss: {:.4f}".format(avg_train_loss))

        return avg_train_loss

    def test(self):
        self.model.eval()
        psnr = 0
        test_loss = 0
        with torch.no_grad():
            for index, (img, target) in enumerate(self.test_loader):
                if img.shape != target.shape:
                    img_BICUBIC = self.convert_BICUBIC(img)
                    target = self.convert_same(img_BICUBIC, target)
                else:
                    img_BICUBIC = img
                img_BICUBIC, target = img_BICUBIC.to(self.device), target.to(self.device)

                _, output = self.model(img_BICUBIC)
                output = output.clamp(0.0, 1.0)

                loss = self.criterion(output, target)
                test_loss += loss.item()
                psnr += self.psrn(loss.item())
                progress_bar(index, len(self.test_loader),
                             'Loss: %.4f, PSNR: %.4f' % (test_loss / (index + 1), psnr / (index + 1)))

        avg_psnr = psnr / len(self.test_loader)
        print("    Average PSNR: {:.4f} dB".format(avg_psnr))
        return avg_psnr

    def run(self):
        self.build_model()
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            print('\n===> alpha={}, Epoch {} starts:'.format(self.loss_alpha, epoch))
            self.loss_alpha = max(0.0, self.loss_alpha - self.loss_alpha_decay)
            avg_train_loss = self.train()
            avg_psnr = self.test()
            self.scheduler.step(self.metric)
            self.logger.info("Epoch [{}/{}]: loss={} PSNR={}".format(epoch, self.epochs + self.start_epoch,
                                                                     avg_train_loss, avg_psnr))

            if avg_psnr > self.best_quality:
                self.best_quality = avg_psnr
                self.save_model(epoch, avg_psnr)
