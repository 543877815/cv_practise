# reference: https://github.com/icpm/super-resolution
import argparse
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Pytorch super resolution example')
    # cuda-configuration
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda')

    # hyper-parameters
    parser.add_argument('--training_batch_size', type=int, default=16, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use')

    # models configuration
    parser.add_argument('--upscaleFactor', '-uf', type=int, default=3, help='super resolution upscale factor')
    parser.add_argument('--model', '-m', type=str, default='srcnn', help='models that going to use')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

    # data configuration
    parser.add_argument('--dataset', type=str, default='customize', help='data that going to use')
    parser.add_argument('--single_channel', action='store_true', help='whether to use specific channel')
    parser.add_argument('--num_workers', type=int, default=1, help='number of worker for data loader')

    parser.add_argument('--use_h5py', action='store_true', help='whether to use .h5 file as data input')
    parser.add_argument('--h5py_input', type=str, default='h5py_input', help='.h5 file data for training')
    parser.add_argument('--train_LR_dir', type=str, default='train_LR_dir', help='low resolution data for training')
    parser.add_argument('--train_HR_dir', type=str, default='train_HR_dir', help='high resolution data for training')
    parser.add_argument('--test_LR_dir', type=str, default='val_LR_dir', help='low resolution data for validation')
    parser.add_argument('--test_HR_dir', type=str, default='val_HR_dir', help='high resolution data for validation')

    parser.add_argument('--use_aug', action='store_true', help='whether to use data augmentation')
    parser.add_argument('--color', type=str, default='RGB', help='color space to use, RGB/YCbCr')

    args = parser.parse_args()

    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    # data/models/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir, log_dir = get_platform_path()
    data_train_dir, data_test_dir, data_val_dir = data_dir, data_dir, data_dir
    train_LR_dir, train_HR_dir, test_LR_dir, test_HR_dir = args.train_LR_dir, args.train_HR_dir, args.test_LR_dir, \
                                                           args.test_HR_dir
    # data preparing
    print("==> Preparing data..")
    dataset = args.dataset
    if dataset.lower() == 'customize':
        train_LR_dir = train_LR_dir
        train_HR_dir = train_HR_dir
    else:
        if dataset.lower() == 'bsd300' or dataset.lower() == 'bsds300':
            data_dir = BSD300()
            data_train_dir = data_dir + '/train'
            data_test_dir = data_dir + '/test'
        elif dataset.lower() == 'bsd500' or dataset.lower() == 'bsds500':
            data_dir = BSDS500()
            data_train_dir = data_dir + '/train'
            data_test_dir = data_dir + '/test'
            data_val_dir = data_dir + '/val'
        elif dataset.lower() == '91images':
            data_dir = images91()
            data_train_dir = data_dir
            data_test_dir = data_dir + '/X2'
        elif dataset.lower() == 'urban100':
            data_dir = Urban100()
            data_train_dir = data_dir + '/HR'
            data_test_dir = data_dir
        elif dataset.lower() == 'set5':
            data_dir = Set5()
            data_train_dir = data_dir + '/HR'
            data_test_dir = data_dir
        elif dataset.lower() == 'set14':
            data_dir = Set14()
            data_train_dir = data_dir + '/HR'
            data_test_dir = data_dir
        elif dataset.lower() == 'b100':
            data_dir = B100()
            data_train_dir = data_dir + '/HR'
            data_test_dir = data_dir
        elif dataset.lower() == 'manga109':
            data_dir = Manga109() + '/HR'
            data_train_dir = data_dir
            data_test_dir = data_dir
        else:
            raise Exception("the dataset does not support")

    # data augmentation
    if args.use_aug:
        upscale_factor = args.upscaleFactor
        crop_size = 256 - (256 % upscale_factor)
        img_transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Resize(crop_size // upscale_factor, interpolation=Image.BICUBIC),
            transforms.ToTensor(),
        ])

        target_transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.ToTensor()
        ])

        train_set = DatasetFromOneFolder(image_dir=data_train_dir, transform=img_transform,
                                         target_transform=target_transform,
                                         config=args)
        test_set = DatasetFromOneFolder(image_dir=data_test_dir, transform=img_transform,
                                        target_transform=target_transform,
                                        config=args)
    else:
        img_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        target_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        if args.use_h5py:
            train_set = TrainDataset(args.h5py_input, transform=img_transform, target_transform=target_transform)
        else:
            train_set = DatasetFromTwoFolder(LR_dir=train_LR_dir, HR_dir=train_HR_dir, transform=img_transform,
                                             target_transform=target_transform,
                                             config=args)

        test_set = DatasetFromTwoFolder(LR_dir=test_LR_dir, HR_dir=test_HR_dir, transform=img_transform,
                                        target_transform=target_transform,
                                        config=args)

    train_loader = DataLoader(dataset=train_set, batch_size=args.training_batch_size, shuffle=True, pin_memory=True,
                              num_workers=args.num_workers)
    test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=False)

    if args.model.lower() == 'srcnn':
        model = SRCNNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'fsrcnn':
        model = FSRCNNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'vdsr':
        model = VDSRTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'espcn':
        model = ESPCNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'drcn':
        model = DRCNTrainer(args, train_loader, test_loader)
    elif args.model.lower() == 'drrn':
        model = DRRNTrainer(args, train_loader, test_loader)
    else:
        raise Exception("the model does not exist")

    model.run()
