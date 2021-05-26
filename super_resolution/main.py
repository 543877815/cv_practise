# reference: https://github.com/icpm/super-resolution
import sys
import os

sys.path.append(os.path.abspath('../'))
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel
import torch.multiprocessing as mp
import torch.distributed as dist
from dataset.dataset import *
import torch
from super_resolution.models.SRCNN.solver import SRCNNTrainer
from super_resolution.models.FSRCNN.solver import FSRCNNTrainer
from super_resolution.models.VDSR.solver import VDSRTrainer
from super_resolution.models.ESPCN.solver import ESPCNTrainer
from super_resolution.models.DRRN.solver import DRRNTrainer
from super_resolution.models.DRCN.solver import DRCNTrainer
from attrdict import AttrDict
from options import args
import yaml


def get_config(args):
    with open(args.filename, 'r') as file:
        try:
            config = yaml.safe_load(file)
            for arg in vars(args):
                if args.config_priority == 'args':
                    config[arg] = getattr(args, arg)
                elif arg not in config.keys():
                    config[arg] = getattr(args, arg)
            config = AttrDict(config)
        except yaml.YAMLError as exc:
            config = None
            print(exc)
    return config


def get_dataset(config):
    data_flist = AttrDict(config.data_flist[config.platform])
    # data/models/checkpoint in different platform
    train_LR_dir, train_HR_dir, test_LR_dir, test_HR_dir = data_flist.train_LR_dir, data_flist.train_HR_dir, \
                                                           data_flist.test_LR_dir, data_flist.test_HR_dir

    # data transform
    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # data preparing
    print("===> Preparing data..")
    if config.use_h5py:
        train_set = DatasetFromH5py(h5_file=data_flist.h5py_input, transform=img_transform,
                                    target_transform=target_transform)
        assert len(train_set), 'No file found at {}'.format(data_flist.h5py_input)
    else:
        dataset = config.dataset
        if dataset.lower() == 'customize':
            if config.buildRawData:
                assert (data_flist.origin_HR_dir and data_flist.train_HR_dir and data_flist.train_LR_dir) is not None, \
                    'origin_HR_dir, train_HR_dir and train_HR_dir should exist when using dataset="customize".'
                buildRawData(origin_HR_dir=data_flist.origin_HR_dir, train_HR_dir=data_flist.train_HR_dir,
                             train_LR_dir=data_flist.train_LR_dir, config=config)
            train_LR_dir = train_LR_dir
            train_HR_dir = train_HR_dir
        else:
            if dataset.lower() == 'bsd300' or dataset.lower() == 'bsds300':
                train_LR_dir, train_HR_dir = BSD300(config)
            elif dataset.lower() == 'bsd500' or dataset.lower() == 'bsds500':
                train_LR_dir, train_HR_dir = BSDS500(config)
            elif dataset.lower() == '91-images':
                train_LR_dir, train_HR_dir = images91(config)
            elif dataset.lower() == 'urban100':
                train_LR_dir, train_HR_dir = Urban100(config)
            elif dataset.lower() == 'set5':
                train_LR_dir, train_HR_dir = Set5(config)
            elif dataset.lower() == 'set14':
                train_LR_dir, train_HR_dir = Set14(config)
            elif dataset.lower() == 'b100':
                train_LR_dir, train_HR_dir = B100(config)
            elif dataset.lower() == 'manga109':
                train_LR_dir, train_HR_dir = Manga109(config)
            elif dataset.lower() == 'div2k':
                data_dir = DIV2K() + '/HR'
            elif dataset.lower() == 'celeb':
                pass
            else:
                raise Exception("the dataset does not support, dataset only support [bsd300, bsd500, "
                                "91-images, urban100, set5, set14, b100, manga109, div2k, celebA].")
        train_set = DatasetFromTwoFolder(LR_dir=train_LR_dir, HR_dir=train_HR_dir, train=True, transform=img_transform,
                                         target_transform=target_transform, config=config)
        assert len(train_set), 'No images found at {} or {}'.format(train_LR_dir, train_HR_dir)

    test_set = DatasetFromTwoFolder(LR_dir=test_LR_dir, HR_dir=test_HR_dir, transform=img_transform,
                                    target_transform=target_transform, config=config)
    assert len(test_set), 'No images found at {} or {}'.format(test_LR_dir, test_HR_dir)
    return train_set, test_set


def get_trainer(config, train_loader, test_loader, device=None):
    # load model
    model_name = config.model
    if model_name.lower() == 'srcnn':
        model = SRCNNTrainer(config, train_loader, test_loader, device)
    elif model_name.lower() == 'fsrcnn':
        model = FSRCNNTrainer(config, train_loader, test_loader, device)
    elif model_name.lower() == 'vdsr':
        model = VDSRTrainer(config, train_loader, test_loader, device)
    elif model_name.lower() == 'espcn':
        model = ESPCNTrainer(config, train_loader, test_loader, device)
    elif model_name.lower() == 'drcn':
        model = DRCNTrainer(config, train_loader, test_loader, device)
    elif model_name.lower() == 'drrn':
        model = DRRNTrainer(config, train_loader, test_loader, device)
    elif model_name.lower() == 'lapsrn':
        # model = LapSRNTrainer(config, train_loader, test_loader)
        pass
    elif model_name.lower() == 'lapsrn-gan':
        # model = LapSRN_GANTrainer(config, train_loader, test_loader)
        pass
    elif model_name.lower() == 'edsr':
        model = None
    else:
        raise Exception("the model does not exist, model only support [srcnn, fsrcnn, "
                        "vdsr, espcn, drcn, drrn, lapsrn, lapsrn-gan, edsr]")
    return model


def main():
    # get configuration
    configs = get_config(args)

    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
    if not configs.distributed:
        device = torch.device(
            "cuda:{}".format(configs.gpu[0]) if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    else:
        device = torch.device("cuda", args.local_rank)
    # get dataset
    train_set, test_set = get_dataset(configs)

    sampler = None
    local_rank = None
    if configs.distributed:
        torch.distributed.init_process_group(backend='nccl')
        local_rank = torch.distributed.get_rank()
        torch.cuda.set_device(local_rank)
        configs.device = torch.device("cuda", local_rank)
        sampler = DistributedSampler(dataset=train_set, shuffle=True)

    train_loader = DataLoader(dataset=train_set, batch_size=configs.training_batch_size, shuffle=sampler is None,
                              pin_memory=True, num_workers=configs.num_workers, drop_last=False, sampler=sampler)
    test_loader = DataLoader(dataset=test_set, batch_size=configs.test_batch_size, shuffle=False)

    # get model
    trainer = get_trainer(configs, train_loader, test_loader, device)
    if configs.distributed and len(configs.gpu) > 1:
        # print(args.rank, args.world_size, args.local_rank, configs.gpu)
        # assert args.rank is None and args.world_size is None, \
        #     'When --distributed is enabled (default) the rank and ' + \
        #     'world size can not be given as this is set up automatically. ' + \
        #     'Use --distributed 0 to disable automatic setup of distributed training.'
        trainer.model = torch.nn.parallel.DistributedDataParallel(trainer.model, output_device=args.local_rank,
                                                                  device_ids=[args.local_rank])
    trainer.run()
    # mp.spawn(run_distributed, nprocs=len(args.gpu), args=(args,))


if __name__ == '__main__':
    main()
