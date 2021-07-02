# reference:
# https://github.com/icpm/super-resolution
# https://github.com/twtygqyy/pytorch-vdsr
import sys
import os

sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('../../../'))
from math import log10
import torch
import torch.backends.cudnn as cudnn
import os
from super_resolution.experiment.VDSR import VDSR, ResNet, BasicBlock, mapping
from utils import progress_bar, get_platform_path, get_logger, shave
from torchvision.transforms import transforms
from PIL import Image
from torchvision import utils as vutils
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict
import numpy as np
from torch.autograd import Variable


class VSDRBasic(object):
    def __init__(self, config, device=None):
        super(VSDRBasic, self).__init__()
        self.CUDA = torch.cuda.is_available()
        self.device = device

        # models configuration
        self.model = None
        self.classifier = None
        self.mapping = None
        self.color_space = config.color_space
        self.num_filter = config.num_filter
        self.num_classes = config.num_classes
        self.num_residuals = config.num_residuals
        self.num_channels = config.num_channels
        self.add_channels = config.add_channels
        self.upscale_factor = config.upscaleFactor
        self.test_upscaleFactor = config.test_upscaleFactor
        self.model_name = "{}-{}x".format(config.model, self.upscale_factor)

        # checkpoint configuration
        self.resume = config.resume
        self.checkpoint_name = "{}.pth".format(self.model_name)
        self.best_quality = 0
        self.start_epoch = 1
        self.epochs = config.epochs
        self.checkpoint_interval = config.checkpoint_interval

        # logger configuration
        _, _, _, log_dir = get_platform_path()
        self.logger = get_logger("{}/{}.log".format(log_dir, self.model_name))
        self.logger.info(config)

        # tensorboard writer
        self.writer = SummaryWriter(config.tensorboard_log_dir)
        self.tensorboard_image_interval = config.tensorboard_image_interval
        self.tensorboard_draw_model = config.tensorboard_draw_model
        self.tensorboard_input = config.tensorboard_input
        self.tensorboard_image_size = config.tensorboard_image_size
        self.tensorboard_image_sample = config.tensorboard_image_sample

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
        self.model.load_state_dict(new_state_dict)
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


class VDSRTester(VSDRBasic):
    def __init__(self, config, test_loader=None):
        super(VDSRTester, self).__init__(config)
        assert (config.resume is True)
        data_dir, _, _, _ = get_platform_path()
        # resolve configuration
        self.output = data_dir + config.output
        self.test_loader = test_loader
        self.criterion = None
        self.build_model()

    def build_model(self):
        self.model = VDSR(num_channels=self.num_channels, num_filter=self.num_filter,
                          num_residuals=self.num_residuals).to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.load_model()
        if self.CUDA:
            cudnn.benchmark = True
            self.criterion.cuda()

    def run(self):
        self.model.eval()
        with torch.no_grad():
            for index, (img, filename) in enumerate(self.test_loader):
                img = img.to(self.device)
                # full RGB/YCrCb
                if self.num_channels == 3:
                    output = self.model(img).clamp(0.0, 1.0).cpu()
                # y
                elif self.num_channels == 1:
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


class VDSRTrainer(VSDRBasic):
    def __init__(self, config, train_loader=None, test_loader=None, device=None):
        super(VDSRTrainer, self).__init__(config, device)

        # parameters configuration
        self.criterion = None
        self.infoLoss = None
        self.optimizer = None
        self.scheduler = None
        self.lr = config.lr
        self.seed = config.seed
        self.clip = config.clip
        self.momentum = config.momentum
        self.scheduler_gamma = config.scheduler_gamma
        self.milestones = config.milestones
        self.weight_decay = config.weight_decay

        # data loader
        self.train_loader = train_loader
        self.test_loader = test_loader

        # models init
        self.build_model()

    def build_model(self):
        self.model = VDSR(num_channels=self.num_channels + self.add_channels, num_filter=self.num_filter,
                          num_residuals=self.num_residuals).to(self.device)
        self.classifier = ResNet(BasicBlock, [1, 1, 1, 1], num_classes=self.num_classes).to(self.device)
        self.mapping = mapping(add_channels=self.add_channels, num_classes=self.num_classes).to(self.device)
        if self.resume:
            self.load_model()
        # else:
        #     self.models.weight_init()

        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.infoLoss = torch.nn.CrossEntropyLoss()  # 离散损失
        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.criterion.to(self.device)
            self.infoLoss.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.weight_decay,
                                         weight_decay=self.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=self.milestones,
                                                              gamma=self.scheduler_gamma)

    def save_model(self, epoch, avg_psnr, name):
        _, _, checkpoint_dir, _ = get_platform_path()
        model_out_path = '{}/{}'.format(checkpoint_dir, name)
        state = {
            'net': self.model.state_dict(),
            'psnr': avg_psnr,
            'epoch': epoch
        }
        torch.save(state, model_out_path)
        self.logger.info("checkpoint saved to {}".format(model_out_path))

    def to_categorical(self, y, num_columns):
        y_cat = np.zeros((y.shape[0], num_columns))
        y_cat[range(y.shape[0]), y] = 1.0
        return Variable(torch.FloatTensor(y_cat))

    def train(self):
        self.model.train()
        train_loss = 0
        for index, (label, img, target) in enumerate(self.train_loader):

            label = self.to_categorical(y=label.numpy(), num_columns=self.num_classes).to(self.device)
            code = self.mapping(label)
            batch_size, channel, width, height = img.shape
            code = code.reshape([batch_size, self.add_channels, 1, 1])
            add_feature = code.repeat(1, 1, width, height)

            assert img.shape == target.shape, 'the shape of input is not equal to the shape of output'
            img, target = img.to(self.device), target.to(self.device)

            output = self.model(img, add_feature)

            resume_code = self.classifier(output)
            loss1 = self.infoLoss(resume_code.float(), label.float())
            loss2 = self.criterion(output, target)

            print(loss1, loss2)
            loss = loss1 + loss2
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
            self.optimizer.step()
            train_loss += loss.item()
            if not self.distributed or self.local_rank == 0:
                progress_bar(index, len(self.train_loader), 'Loss: %.4f' % (train_loss / (index + 1)))

        avg_train_loss = train_loss / len(self.train_loader)
        print("    Average Loss: {:.4f}".format(avg_train_loss))

        return avg_train_loss

    def test(self):
        self.model.eval()
        psnr = 0
        # random sample output to tensorboard
        save_inputs, save_outputs, save_targets = [], [], []
        with torch.no_grad():
            for index, (img, target) in enumerate(self.test_loader):
                assert img.shape == target.shape, 'the shape of input is not equal to the shape of output'
                img, target = img.to(self.device), target.to(self.device)
                output = self.model(img).clamp(0.0, 1.0)
                output, target = shave(output, target, self.test_upscaleFactor)
                loss = self.criterion(output, target)
                psnr += self.psrn(loss.item() / target.shape[2] / target.shape[3])
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
        for epoch in range(self.start_epoch, self.epochs + self.start_epoch):
            print('\n===> Epoch {} starts:'.format(epoch))
            avg_train_loss = self.train()
            avg_psnr, save_input, save_output, save_target = self.test()
            self.scheduler.step()

            if not self.distributed or self.local_rank == 0:

                # save to logger
                self.logger.info(
                    "Epoch [{}/{}]: lr={:.6f} loss={:.6f} PSNR={:.6f}".format(epoch, self.epochs + self.start_epoch,
                                                                              self.optimizer.param_groups[0]['lr'],
                                                                              avg_train_loss, avg_psnr))

                # save best models
                if avg_psnr > self.best_quality:
                    self.best_quality = avg_psnr
                    self.save_model(epoch, avg_psnr, self.checkpoint_name)

                # save interval models
                if epoch % self.checkpoint_interval == 0:
                    name = self.checkpoint_name.replace('.pth', '_{}.pth'.format(epoch))
                    self.save_model(epoch, avg_psnr, name)

                # tensorboard scalars
                self.writer.add_scalar('train_loss', avg_train_loss, epoch)
                self.writer.add_scalar('psnr', avg_psnr, epoch)

                # tensorboard graph
                if epoch == 0 and self.tensorboard_draw_model and \
                        len(save_input) > 0 and len(save_target) > 0:
                    self.writer.add_graph(model=self.model, input_to_model=[save_input[0]])

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
