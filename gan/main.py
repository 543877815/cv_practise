import os
import sys

sys.path.append(os.path.abspath('../'))

import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from gan.models.GAN import GAN
from gan.models.WGAN.WGAN import WGAN
from gan.models.DCGAN.DCGAN import DCGAN
from gan.models.CGAN.CGAN import CGAN
from gan.models.RaGAN import RaGAN
from gan.models.InfoGAN import InfoGAN
from gan.models.LSGAN import LSGAN
from options import args
from utils import get_config
from attrdict import AttrDict


def get_dataset(config):
    # data/models/checkpoint in different platform
    data_flist = AttrDict(config.data_flist[config.platform])
    train_dir = data_flist.train_dir

    # data preparing
    print("==> Preparing data..")
    if config.dataset.lower() == 'mnist':
        transform_train = transforms.Compose([
            transforms.Resize(config.img_size), transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        train_set = torchvision.datasets.MNIST(root=train_dir, train=True, download=True, transform=transform_train)
    else:
        raise Exception("the dataset does not exist")

    return train_set


def get_trainer(config, dataloader, device=None):
    if configs.model.lower() == 'gan':
        model = GAN(config=config, dataloader=dataloader, device=device)
    elif configs.model.lower() == 'wgan':
        model = WGAN(config=config, dataloader=dataloader, device=device)
    elif configs.model.lower() == 'dcgan':
        model = DCGAN(config=config, dataloader=dataloader, device=device)
    elif configs.model.lower() == 'cgan':
        model = CGAN(config=config, dataloader=dataloader, device=device)
    elif configs.model.lower() == 'infogan':
        model = InfoGAN(config=config, dataloader=dataloader, device=device)
    elif configs.model.lower() == 'ragan':
        model = RaGAN(config=config, dataloader=dataloader, device=device)
    elif configs.model.lower() == 'lsgan':
        model = LSGAN(config=config, dataloader=dataloader, device=device)
    else:
        raise Exception("the models does not exist")

    return model


if __name__ == '__main__':
    # get configuration
    configs = get_config(args)
    img_shape = (args.channels, args.img_size, args.img_size)

    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
    if not configs.distributed:
        device = torch.device(
            "cuda:{}".format(configs.gpu[0]) if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device("cuda", args.local_rank)

    # dataset
    train_set = get_dataset(configs)
    sampler = None
    local_rank = None
    if configs.distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        configs.device = torch.device("cuda", local_rank)
        sampler = DistributedSampler(dataset=train_set, shuffle=True)

    dataloader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        num_workers=args.num_workers,
        sampler=sampler,
        drop_last=False, )

    # models
    trainer = get_trainer(configs, dataloader, device)
    if configs.distributed and len(configs.gpu) > 1:
        # print(args.rank, args.world_size, args.local_rank, configs.gpu)
        # assert args.rank is None and args.world_size is None, \
        #     'When --distributed is enabled (default) the rank and ' + \
        #     'world size can not be given as this is set up automatically. ' + \
        #     'Use --distributed 0 to disable automatic setup of distributed training.'
        trainer.model = torch.nn.parallel.DistributedDataParallel(trainer.model, output_device=args.local_rank,
                                                                  device_ids=[args.local_rank])
    trainer.train()
