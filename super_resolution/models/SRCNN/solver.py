# https://github.com/icpm/super-resolution
from math import log10

import torch
import torch.backends.cudnn as cudnn
import os
from .model import SRCNN
from utils import progress_bar, get_platform_path, get_logger
from torchvision.transforms import transforms
from PIL import Image
from torchvision import utils as vutils
import matplotlib.pyplot as plt


class SRCNNBasic(object):
    def __init__(self, config, device=None):
        super(SRCNNBasic, self).__init__()
        self.CUDA = torch.cuda.is_available()
        if device is None:
            self.device = torch.device("cuda" if (config.use_cuda and self.CUDA) else "cpu")

        # model configuration
        self.model = None
        self.color_space = config.color
        self.single_channel = config.single_channel
        self.upscale_factor = config.upscaleFactor
        self.model_name = "SRCNN-{}x".format(self.upscale_factor)

        # checkpoint configuration
        self.resume = config.resume
        self.checkpoint_name = "{}.pth".format(self.model_name)
        self.best_quality = 0
        self.start_epoch = 1

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


class SRCNNTester(SRCNNBasic):
    def __init__(self, config, test_loader=None, device=None):
        super(SRCNNTester, self).__init__(config)
        assert (config.resume is True)

        data_dir, _, _, _ = get_platform_path()
        # resolve configuration
        self.output = data_dir + config.output
        self.test_loader = test_loader
        self.criterion = torch.nn.MSELoss()

    def build_model(self):
        num_channels = 1 if self.single_channel else 3
        self.model = SRCNN(num_channels=num_channels, filter=64).to(self.device)
        self.load_model()
        if self.CUDA:
            cudnn.benchmark = True
            self.criterion.cuda()

    def run(self):
        self.build_model()
        self.model.eval()
        with torch.no_grad():
            for index, (img, filename) in enumerate(self.test_loader):
                img_BICUBIC = self.convert_BICUBIC(img[0])
                img_BICUBIC = img_BICUBIC.to(self.device)
                # full RGB/YCrCb
                if not self.single_channel:
                    output = self.model(img_BICUBIC).clamp(0.0, 1.0).cpu()
                # y
                else:
                    output = self.model(img_BICUBIC).cpu()
                    output = [output, img[1], img[2]]
                output_name = filename[0].replace('LR', 'HR')
                if self.color_space == 'RGB':
                    vutils.save_image(output, self.output + output_name)
                elif self.color_space == 'YCbCr':
                    output = transforms.ToPILImage(mode='YCbCr')(output[0]).convert("RGB")
                    output.save(self.output + output_name)
                print('==> {} is saved to {}'.format(output_name, self.output))


class SRCNNTrainer(SRCNNBasic):
    def __init__(self, config, train_loader=None, test_loader=None, device=None):
        super(SRCNNTrainer, self).__init__(config, device)

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

    def build_model(self):
        num_channels = 1 if self.single_channel else 3
        self.model = SRCNN(num_channels=num_channels, filter=64).to(self.device)
        if self.resume:
            self.load_model()
        else:
            self.model.weight_init()

        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam([
            {'params': self.model.conv1.parameters()},
            {'params': self.model.conv2.parameters()},
            {'params': self.model.conv3.parameters(), 'lr': self.lr * 0.1}
        ], lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[250, 500, 750], gamma=0.5)

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

            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            progress_bar(index, len(self.train_loader), 'Loss: %.4f' % (train_loss / (index + 1)))

        avg_train_loss = train_loss / len(self.train_loader)
        print("    Average Loss: {:.6f}".format(avg_train_loss))
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

                psnr += self.psrn(loss.item())
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
