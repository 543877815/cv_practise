# reference: https://github.com/icpm/super-resolution
import argparse
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from utils import *
from dataset.dataset import *
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch super resolution example')
    # cuda-configuration
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda')

    # hyper-parameters
    parser.add_argument('--training_batch_size', type=int, default=1, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
    parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--seed', type=int, default=123, help='random seed to use')

    # models configuration
    parser.add_argument('--upscaleFactor', '-uf', type=int, default=4, help='super resolution upscale factor')
    parser.add_argument('--models', '-m', type=str, default='srcnn', help='models that going to use')

    # data configuration
    parser.add_argument('--dataset', type=str, default='bsd300', help='data that going to use')

    args = parser.parse_args()

    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    # data/models/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir = get_platform_path()
    data_train_dir, data_test_dir = data_dir, data_dir

    # data preparing
    print("==> Preparing data..")
    dataset = args.dataset
    if dataset == 'bsd300':
        data_dir = BSD300()
        data_train_dir = data_dir + '/train'
        data_test_dir = data_dir + '/test'

    # data augmentation
    upscale_factor = args.upscaleFactor
    crop_size = 256 - (256 % upscale_factor)

    img_transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.Resize(crop_size // upscale_factor),
        transforms.ToTensor(),
    ])

    target_transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()
    ])

    train_set = DatasetFromFolder(image_dir=data_train_dir, transform=img_transform, target_transform=target_transform)
    test_set = DatasetFromFolder(image_dir=data_test_dir, transform=img_transform, target_transform=target_transform)

    train_loader = DataLoader(dataset=train_set, batch_size=args.training_batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=args.test_batch_size, shuffle=False)

    if args.model == 'srcnn':
        model = SRCNNTrainer(args, train_loader, test_loader)

    model.run();
