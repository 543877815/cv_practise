# https://github.com/icpm/super-resolution
from math import log10

import torch
import torch.backends.cudnn as cudnn
import os
from .model import FSRCNN
from utils import progress_bar, get_platform_path
from torchvision.transforms import transforms
from PIL import Image
from torchvision import utils as vutils


class FSRCNNBasic(object):
    def __init__(self, config, device=None):
        super(FSRCNNBasic, self).__init__()
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
        self.checkpoint_name = "FSRCNN-{}x.pth".format(self.upscale_factor)
        self.best_quality = 0
        self.start_epoch = 1

    def load_model(self):
        _, _, checkpoint_dir, _ = get_platform_path()
        print('==> Resuming from checkpoint...')
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('{}/{}'.format(checkpoint_dir, self.checkpoint_name))
        self.model.load_state_dict(checkpoint['net'])
        self.best_quality = checkpoint['psnr']
        self.start_epoch = checkpoint['epoch']

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


class FSRCNNTester(FSRCNNBasic):
    def __init__(self, config, test_loader=None, device=None):
        super(FSRCNNTester, self).__init__(config)
        assert (config.resume is True)

        data_dir, _, _, _ = get_platform_path()
        # resolve configuration
        self.output = data_dir + config.output
        self.test_loader = test_loader
        self.criterion = torch.nn.MSELoss()

    def build_model(self):
        num_channels = 1 if self.single_channel else 3
        self.model = FSRCNN(num_channels=num_channels, upscale_factor=self.upscale_factor).to(self.device)
        self.load_model()
        if self.CUDA:
            cudnn.benchmark = True
            self.criterion.cuda()

    def run(self):
        self.build_model()
        self.model.eval()
        with torch.no_grad():
            for index, (img, filename) in enumerate(self.test_loader):
                img = img.to(self.device)
                # full RGB/YCrCb
                if not self.single_channel:
                    output = self.model(img).clamp(0.0, 1.0).cpu()
                # y
                else:
                    output = self.model(img[:, 0, :, :].unsqueeze(1))
                    img[:, 0, :, :].data = output
                    output = img.clamp(0.0, 1.0).cpu()
                assert (len(output.shape) == 4 and output.shape[0] == 1)
                output_name = filename[0].replace('LR', 'HR')
                if self.color_space == 'RGB':
                    vutils.save_image(output, self.output + output_name)
                elif self.color_space == 'YCbCr':
                    output = transforms.ToPILImage(mode='YCbCr')(output[0]).convert("RGB")
                    output.save(self.output + output_name)
                print('==> {} is saved to {}'.format(output_name, self.output))


class FSRCNNTrainer(FSRCNNBasic):
    def __init__(self, config, train_loader=None, test_loader=None, device=None):
        super(FSRCNNTrainer, self).__init__(config, device)

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
        self.model = FSRCNN(num_channels=num_channels, upscale_factor=self.upscale_factor).to(self.device)
        if self.resume:
            self.load_model()
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam([
            {'params': self.model.first_part.parameters()},
            {'params': self.model.mid_part.parameters()},
            {'params': self.model.last_part.parameters(), 'lr': self.lr * 0.1}
        ], lr=self.lr)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[100, 200, 300], gamma=0.5)

    def save_model(self, epoch, avg_psnr):
        _, _, checkpoint_dir, _ = get_platform_path()
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
            img, target = img.to(self.device), target.to(self.device)

            output = self.model(img)
            loss = self.criterion(output, target)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            progress_bar(index, len(self.train_loader), 'Loss: %.4f' % (train_loss / (index + 1)))

        avg_train_loss = train_loss / len(self.train_loader)
        print("    Average Loss: {:.4f}".format(avg_train_loss))

    def test(self):
        self.model.eval()
        psnr = 0
        with torch.no_grad():
            for index, (img, target) in enumerate(self.test_loader):
                img, target = img.to(self.device), target.to(self.device)

                output = self.model(img).clamp(0.0, 1.0)
                if output.shape != target.shape:
                    target = self.convert_same(output, target).to(self.device)
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
            self.train()
            avg_psnr = self.test()
            self.scheduler.step()

            if avg_psnr > self.best_quality:
                self.best_quality = avg_psnr
                self.save_model(epoch, avg_psnr)
