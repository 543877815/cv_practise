from math import log10

import torch
import torch.backends.cudnn as cudnn
import os
from .model import SRCNN
from utils import progress_bar, get_platform_path


class SRCNNTrainer(object):
    def __init__(self, config, train_loader, test_loader, device=None):
        super(SRCNNTrainer, self).__init__()
        self.CUDA = torch.cuda.is_available()
        if device is None:
            self.device = torch.device("cuda" if (config.use_cuda and self.CUDA) else "cpu")

        # model configuration
        self.model = None
        self.lr = config.lr

        self.single_channel = config.single_channel
        self.upscale_factor = config.upscaleFactor

        # checkpoint configuration
        self.resume = config.resume
        self.best_quality = 0
        self.epochs = config.epochs
        self.start_epoch = 1
        self.checkpoint_name = "SRCNN.pth"

        # parameters configuration
        self.criterion = None
        self.optimizer = None
        self.scheduler = None
        self.seed = config.seed

        self.train_loader = train_loader
        self.test_loader = test_loader

    def build_model(self):
        num_channels = 1 if self.single_channel else 3
        self.model = SRCNN(num_channels=num_channels, filter=64, upscale_factor=self.upscale_factor).to(self.device)
        if self.resume:
            self.load_model()

        else:
            self.model.weight_init(mean=0.0, std=0.01)
        self.criterion = torch.nn.MSELoss()
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.cuda()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[50, 75, 100], gamma=0.5)

    def load_model(self):
        _, _, checkpoint_dir = get_platform_path()
        print('==> Resuming from checkpoint...')
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('{}/{}'.format(checkpoint_dir, self.checkpoint_name))
        self.model.load_state_dict(checkpoint['net'])
        self.best_quality = checkpoint['psnr']
        self.start_epoch = checkpoint['epoch']

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
            img, target = img.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            loss = self.criterion(self.model(img), target)
            train_loss += loss.item()
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
                img, target = img.to(self.device), target.to(self.device)
                prediction = self.model(img)
                mse = self.criterion(prediction, target)
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

    @staticmethod
    def psrn(mse):
        return 10 * log10(1 / mse.item())
