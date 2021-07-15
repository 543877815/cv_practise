import torch
from torch.autograd import Variable
import numpy as np
from utils import get_platform_path
from .model import Generator, Discriminator, weights_init_normal
import torch.backends.cudnn as cudnn
from torchvision.utils import save_image
import os
from gan.common import LongTensor, FloatTensor
import itertools


class InfoGAN(object):

    def __init__(self, config, dataloader=None, device=None):
        super(InfoGAN, self).__init__()

        # hardware
        self.CUDA = torch.cuda.is_available()
        self.device = device

        # data configuration
        self.dataloader = dataloader

        # models configuration
        self.latent_dim = config.latent_dim
        self.img_size = config.img_size
        self.n_classes = config.n_classes
        self.model_name = config.model
        self.channels = config.channels
        self.code_dim = config.code_dim

        # experiment configuration
        self.epochs = config.epoch
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.generator = None
        self.optimizer_G = None
        self.discriminator = None
        self.optimizer_D = None
        self.adversarial_loss = None
        self.categorical_loss = None
        self.continuous_loss = None
        self.seed = 123

        # Loss weights
        self.lambda_cat = 1
        self.lambda_con = 0.1

        # Static generator inputs for sampling
        self.static_z = Variable(FloatTensor(np.zeros((self.n_classes ** 2, self.latent_dim))))
        self.static_label = self.to_categorical(
            np.array([num for _ in range(self.n_classes) for num in range(self.n_classes)]), num_columns=self.n_classes
        )  # one hot 从 0-9 [100, 10]
        self.static_code = Variable(FloatTensor(np.zeros((self.n_classes ** 2, self.code_dim)))) # [100,2]

        # checkpoint
        self.sample_interval = config.sample_interval

        # build model
        self.build_model()

    def build_model(self):
        self.generator = Generator(n_classes=self.n_classes, latent_dim=self.latent_dim, img_size=self.img_size,
                                   channels=self.channels, code_dim=self.code_dim)
        self.generator.apply(weights_init_normal)
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.discriminator = Discriminator(n_classes=self.n_classes, img_size=self.img_size, channels=self.channels,
                                           code_dim=self.code_dim)
        self.discriminator.apply(weights_init_normal)
        self.optimizer_D = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=(self.beta1, self.beta2))
        self.optimizer_info = torch.optim.Adam(
            itertools.chain(self.generator.parameters(), self.discriminator.parameters()), lr=self.lr,
            betas=(self.beta1, self.beta2))
        self.adversarial_loss = torch.nn.MSELoss()
        self.categorical_loss = torch.nn.CrossEntropyLoss()    # 非连续
        self.continuous_loss = torch.nn.MSELoss()              # 连续

        torch.manual_seed(self.seed)

        if self.CUDA:
            torch.cuda.manual_seed(self.seed)
            cudnn.benchmark = True
            self.generator.cuda()
            self.discriminator.cuda()
            self.adversarial_loss.cuda()
            self.categorical_loss.cuda()
            self.continuous_loss.cuda()

    def train(self):
        # ----------
        #  Training
        # ----------
        for epoch in range(self.epochs):
            for i, (imgs, labels) in enumerate(self.dataloader):

                batch_size = imgs.shape[0]

                # Adversarial ground truths
                valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
                fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

                # Configure input
                real_imgs = Variable(imgs.type(FloatTensor))
                labels = self.to_categorical(labels.numpy(), num_columns=self.n_classes)

                # -----------------
                #  Train Generator
                # -----------------

                self.optimizer_G.zero_grad()

                # Sample noise and labels as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                label_input = self.to_categorical(np.random.randint(0, self.n_classes, batch_size),
                                                  num_columns=self.n_classes)
                code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, self.code_dim))))

                # Generate a batch of images
                gen_imgs = self.generator(z, label_input, code_input)

                # Loss measures generator's ability to fool the discriminator
                validity, _, _ = self.discriminator(gen_imgs)
                g_loss = self.adversarial_loss(validity, valid)

                g_loss.backward()
                self.optimizer_G.step()

                # ---------------------
                #  Train Discriminator
                # ---------------------

                self.optimizer_D.zero_grad()

                # Loss for real images
                real_pred, _, _ = self.discriminator(real_imgs)
                d_real_loss = self.adversarial_loss(real_pred, valid)

                # Loss for fake images
                fake_pred, _, _ = self.discriminator(gen_imgs.detach())
                d_fake_loss = self.adversarial_loss(fake_pred, fake)

                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2

                d_loss.backward()
                self.optimizer_D.step()

                # ------------------
                # Information Loss
                # ------------------

                self.optimizer_info.zero_grad()

                # Sample labels
                sampled_labels = np.random.randint(0, self.n_classes, batch_size)

                # Ground truth labels
                gt_labels = Variable(LongTensor(sampled_labels), requires_grad=False)

                # Sample noise, labels and code as generator input
                z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, self.latent_dim))))
                label_input = self.to_categorical(sampled_labels, num_columns=self.n_classes)
                code_input = Variable(FloatTensor(np.random.uniform(-1, 1, (batch_size, self.code_dim))))

                gen_imgs = self.generator(z, label_input, code_input)
                _, pred_label, pred_code = self.discriminator(gen_imgs)

                info_loss = self.lambda_cat * self.categorical_loss(pred_label, gt_labels) + self.lambda_con * \
                            self.continuous_loss(pred_code, code_input)

                info_loss.backward()
                self.optimizer_info.step()

                # --------------
                # Log Progress
                # --------------

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, self.epochs, i, len(self.dataloader), d_loss.item(), g_loss.item())
                )

                batches_done = epoch * len(self.dataloader) + i
                if batches_done % self.sample_interval == 0:
                    self.sample_image(n_row=10, batches_done=batches_done)

    def sample_image(self, n_row, batches_done):
        data_dir, _, _, _ = get_platform_path()
        save_dir = '{}/{}'.format(data_dir, self.model_name)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
            os.mkdir(os.path.join(save_dir, 'static'))
            os.mkdir(os.path.join(save_dir, 'varying_c1'))
            os.mkdir(os.path.join(save_dir, 'varying_c2'))

        """Saves a grid of generated digits ranging from 0 to n_classes"""
        # Static sample
        z = Variable(FloatTensor(np.random.normal(0, 1, (n_row ** 2, self.latent_dim))))
        static_sample = self.generator(z, self.static_label, self.static_code)
        save_image(static_sample.data, "{}/static/{}.png".format(save_dir, batches_done), nrow=n_row, normalize=True)

        # Get varied c1 and c2
        zeros = np.zeros((n_row ** 2, 1))
        # np.linspace(-first, first, n_row) =
        # [-first.        , -0.77777778, -0.55555556, -0.33333333, -0.11111111,
        # 0.11111111,  0.33333333,  0.55555556,  0.77777778,  first.        ]
        c_varied = np.repeat(np.linspace(-1, 1, n_row)[:, np.newaxis], n_row, 0)  # (100, first)
        c1 = Variable(FloatTensor(np.concatenate((c_varied, zeros), -1)))         # (100, 2)
        c2 = Variable(FloatTensor(np.concatenate((zeros, c_varied), -1)))         # (100, 2)
        sample1 = self.generator(self.static_z, self.static_label, c1)
        sample2 = self.generator(self.static_z, self.static_label, c2)
        save_image(sample1.data, "{}/varying_c1/{}.png".format(save_dir, batches_done), nrow=n_row, normalize=True)
        save_image(sample2.data, "{}/varying_c2/{}.png".format(save_dir, batches_done), nrow=n_row, normalize=True)

    def to_categorical(self, y, num_columns):
        """Returns one-hot encoded Variable"""
        y_cat = np.zeros((y.shape[0], num_columns))
        y_cat[range(y.shape[0]), y] = 1.0

        return Variable(FloatTensor(y_cat))
