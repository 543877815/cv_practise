# https://github.com/icpm/super-resolution
from math import log10

import torch
import torch.backends.cudnn as cudnn
import os
from .model import VDSR
from utils import progress_bar, get_platform_path
from torchvision.transforms import transforms
from PIL import Image
from torchvision import utils as vutils


class VSDRBasic(object):
    def __init__(self, config, device=None):
        super(VSDRBasic, self).__init__()
        self.CUDA = torch.cuda.is_available()
        if device is None:
            self.device = torch.device("cuda" if (config.use_cuda and self.CUDA) else "cpu")

        # model configuration
        self.model = None
        self.color_space = config.color
        self.single_channel = config.single_channel
        self.upscale_factor = config.upscaleFactor

        # checkpoint configuration
        self.resume = config.resume
        self.checkpoint_name = "VDSR-{}x.pth".format(self.upscale_factor)
        self.best_quality = 0
        self.start_epoch = 1

    def load_model(self):
        _, _, checkpoint_dir = get_platform_path()
        print('==> Resuming from checkpoint...')
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('{}/{}'.format(checkpoint_dir, self.checkpoint_name))
        self.model.load_state_dict(checkpoint['net'])
        self.best_quality = checkpoint['psnr']
        self.start_epoch = checkpoint['epoch']

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

    @staticmethod
    def psrn(mse):
        return 10 * log10(1 / mse.item())


class VDSRTester(VSDRBasic):
    def __init__(self, config, test_loader=None, device=None):
        super(VDSRTester, self).__init__(config)
        assert (config.resume is True)

        data_dir, _, _ = get_platform_path()
        # resolve configuration
        self.output = data_dir + config.output
        self.test_loader = test_loader
        self.criterion = torch.nn.MSELoss()

    def build_model(self):
        num_channels = 1 if self.single_channel else 3
        self.model = VDSR(num_channels=num_channels, base_channels=64, num_residuals=20).to(self.device)
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
                if not self.single_channel:
                    output = self.model(img_BICUBIC).clamp(0.0, 1.0).cpu()
                # y
                else:
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
    def __init__(self, config, train_loader=None, test_loader=None, device=None):
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

    def build_model(self):
        num_channels = 1 if self.single_channel else 3
        self.model = VDSR(num_channels=num_channels, base_channels=64, num_residuals=20).to(self.device)
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

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[500, 750, 1000], gamma=0.5)

    def save_model(self, epoch, avg_psnr):
        _, _, checkpoint_dir = get_platform_path()
        model_out_path = '{}/{}'.format(checkpoint_dir, self.checkpoint_name)
        state = {
            'net': self.model.state_dict(),
            'psnr': avg_psnr,
            'epoch': epoch
        }
        torch.save(state, model_out_path)
        print("checkpoint saved to {}".format(model_out_path))

    def train(self):
        self.model.train()
        train_loss = 0
        for index, (img, target) in enumerate(self.train_loader):
            img_BICUBIC = self.convert_BICUBIC(img)
            img_BICUBIC, target = img_BICUBIC.to(self.device), target.to(self.device)
            # full RGB/YCrCb
            if not self.single_channel:
                output = self.model(img_BICUBIC)
                loss = self.criterion(output, target)
            # y
            else:
                output = self.model(img_BICUBIC[:, 0, :, :].unsqueeze(1))
                loss = self.criterion(output, target[:, 0, :, :].unsqueeze(1))
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            progress_bar(index, len(self.train_loader), 'Loss: %.4f' % (train_loss / (index + 1)))

        avg_train_loss = train_loss / len(self.train_loader)
        print("    Average Loss: {:.4f}".format(avg_train_loss))

    def test(self):
        self.model.eval()
        psnr = 0
        with torch.no_grad():
            for index, (img, target) in enumerate(self.test_loader):
                img_BICUBIC = self.convert_BICUBIC(img)
                img_BICUBIC, target = img_BICUBIC.to(self.device), target.to(self.device)
                # full RGB/YCrCb
                if not self.single_channel:
                    output = self.model(img_BICUBIC)
                    mse = self.criterion(output, target)
                # y
                else:
                    output = self.model(img_BICUBIC[:, 0, :, :].unsqueeze(1))
                    img_BICUBIC[:, 0, :, :].data = output
                    mse = self.criterion(img_BICUBIC, target)
                psnr += self.psrn(mse)
                progress_bar(index, len(self.test_loader), 'PSNR: %.4f' % (psnr / (index + 1)))

        avg_psnr = psnr / len(self.test_loader)
        print("    Average PSNR: {:.4f} dB".format(avg_psnr))
        return avg_psnr

    def run(self):
        self.build_model()
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            print('\n===> Epoch {} starts:'.format(epoch))
            self.train()
            avg_psnr = self.test()
            self.scheduler.step()

            if avg_psnr > self.best_quality:
                self.best_quality = avg_psnr
                self.save_model(epoch, avg_psnr)