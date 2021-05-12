# reference: https://github.com/icpm/super-resolution
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
    # data/models/checkpoint in different platform
    train_LR_dir, train_HR_dir, test_LR_dir, test_HR_dir = config.train_LR_dir, config.train_HR_dir, \
                                                           config.test_LR_dir, config.test_HR_dir
    # data preparing
    print("===> Preparing data..")
    dataset = config.dataset
    if dataset.lower() == 'customize':
        if config.buildRawData:
            assert (config.Origin_HR_dir and config.train_HR_dir and config.train_LR_dir) is not None, \
                'Origin_HR_dir, train_HR_dir and train_HR_dir should exist when using dataset="customize".'
            buildRawData(Origin_HR_dir=config.Origin_HR_dir, train_HR_dir=config.train_HR_dir,
                         train_LR_dir=config.train_LR_dir, config=config)
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

    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if config.use_h5py:
        train_set = DatasetFromH5py(config.h5py_input, transform=img_transform, target_transform=target_transform)
        assert len(train_set), 'No file found at {}'.format(config.h5py_input)
    else:
        train_set = DatasetFromTwoFolder(LR_dir=train_LR_dir, HR_dir=train_HR_dir, train=True, transform=img_transform,
                                         target_transform=target_transform, config=config)
        assert len(train_set), 'No images found at {} or {}'.format(config.train_LR_dir, train_HR_dir)

    test_set = DatasetFromTwoFolder(LR_dir=test_LR_dir, HR_dir=test_HR_dir, transform=img_transform,
                                    target_transform=target_transform, config=config)
    assert len(test_set), 'No images found at {} or {}'.format(config.test_LR_dir, test_HR_dir)
    return train_set, test_set


def get_model(config, train_loader, test_loader):
    # load model
    model_name = config.model
    if model_name.lower() == 'srcnn':
        model = SRCNNTrainer(config, train_loader, test_loader)
    elif model_name.lower() == 'fsrcnn':
        model = FSRCNNTrainer(config, train_loader, test_loader)
    elif model_name.lower() == 'vdsr':
        model = VDSRTrainer(config, train_loader, test_loader)
    elif model_name.lower() == 'espcn':
        model = ESPCNTrainer(config, train_loader, test_loader)
    elif model_name.lower() == 'drcn':
        model = DRCNTrainer(config, train_loader, test_loader)
    elif model_name.lower() == 'drrn':
        model = DRRNTrainer(config, train_loader, test_loader)
    elif model_name.lower() == 'lapsrn':
        model = LapSRNTrainer(config, train_loader, test_loader)
    elif model_name.lower() == 'lapsrn-gan':
        model = LapSRN_GANTrainer(config, train_loader, test_loader)
    elif model_name.lower() == 'edsr':
        model = None
    else:
        raise Exception("the model does not exist, model only support [srcnn, fsrcnn, "
                        "vdsr, espcn, drcn, drrn, lapsrn, lapsrn-gan, edsr]")
    return model


def run_distributed(model, rank, args):
    args.rank = rank
    args.world_size = len(args.gpu)
    args.gpu = args.gpu[rank]
    args.master_addr = args.master_addr or '127.0.0.1'
    args.master_port = args.master_port or '23456'
    model.run()


def main():
    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    # get configuration
    configs = get_config(args)

    # get dataset
    train_set, test_set = get_dataset(configs)

    sampler = None
    if configs.rank is not None:
        sampler = DistributedSampler(dataset=train_set, shuffle=True)

    train_loader = DataLoader(dataset=train_set, batch_size=configs.training_batch_size, shuffle=sampler is None,
                              pin_memory=True, num_workers=configs.num_workers, drop_last=False, sampler=sampler)
    test_loader = DataLoader(dataset=test_set, batch_size=configs.test_batch_size, shuffle=False)

    # get model
    model = get_model(configs, train_loader, test_loader)

    if len(args.gpu) > 1 and args.distributed:
        assert args.rank is None and args.world_size is None, \
            'When --distributed is enabled (default) the rank and ' + \
            'world size can not be given as this is set up automatically. ' + \
            'Use --distributed 0 to disable automatic setup of distributed training.'

        mp.spawn(run_distributed, nprocs=len(args.gpu), args=(args,))

    else:
        model.run()


if __name__ == '__main__':
    main()
