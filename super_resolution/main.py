# reference: https://github.com/icpm/super-resolution
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils import get_platform_path
from dataset.dataset import *
import torch
from super_resolution.models.SRCNN.solver import SRCNNTrainer
from super_resolution.models.FSRCNN.solver import FSRCNNTrainer
from super_resolution.models.VDSR.solver import VDSRTrainer
from super_resolution.models.ESPCN.solver import ESPCNTrainer
from super_resolution.models.DRRN.solver import DRRNTrainer
from super_resolution.models.DRCN.solver import DRCNTrainer
from options import args
import yaml
from attrdict import AttrDict

if __name__ == '__main__':

    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    # data/models/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir, log_dir = get_platform_path(args)
    data_train_dir, data_test_dir = data_dir, data_dir
    train_LR_dir, train_HR_dir, test_LR_dir, test_HR_dir = args.train_LR_dir, args.train_HR_dir, args.test_LR_dir, \
                                                           args.test_HR_dir

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
            print(exc)

    # data preparing
    print("===> Preparing data..")
    dataset = config.dataset
    if dataset.lower() == 'customize':
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
            raise Exception("the dataset does not support")

    img_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    target_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    if config.use_h5py:
        train_set = DatasetFromH5py(config.h5py_input, transform=img_transform, target_transform=target_transform)
    else:
        train_set = DatasetFromTwoFolder(LR_dir=train_LR_dir, HR_dir=train_HR_dir, train=True, transform=img_transform,
                                         target_transform=target_transform, config=config)

    train_loader = DataLoader(dataset=train_set, batch_size=config.training_batch_size, shuffle=True,
                              pin_memory=True,
                              num_workers=config.num_workers)
    test_set = DatasetFromTwoFolder(LR_dir=test_LR_dir, HR_dir=test_HR_dir, transform=img_transform,
                                    target_transform=target_transform, config=config)
    test_loader = DataLoader(dataset=test_set, batch_size=config.test_batch_size, shuffle=False)

    model_name = config.model
    if model_name.lower() == 'srcnn':
        model = SRCNNTrainer(config, train_loader, test_loader)
    elif model_name.lower() == 'fsrcnn':
        model = FSRCNNTrainer(args, train_loader, test_loader)
    elif model_name.lower() == 'vdsr':
        model = VDSRTrainer(args, train_loader, test_loader)
    elif model_name.lower() == 'espcn':
        model = ESPCNTrainer(args, train_loader, test_loader)
    elif model_name.lower() == 'drcn':
        model = DRCNTrainer(args, train_loader, test_loader)
    elif model_name.lower() == 'drrn':
        model = DRRNTrainer(args, train_loader, test_loader)
    elif model_name.lower() == 'lapsrn':
        model = LapSRNTrainer(args, train_loader, test_loader)
    elif model_name.lower() == 'lapsrn-gan':
        model = LapSRN_GANTrainer(args, train_loader, test_loader)
    elif model_name.lower() == 'edsr':
        model = None
    else:
        raise Exception("the model does not exist")

    model.run()
