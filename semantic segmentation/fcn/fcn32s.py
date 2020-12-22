"""Train FCN32s with PyTorch"""

"""Implemente refer to https://github.com/wkentaro/pytorch-fcn"""




import torchvision
import torch
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn

from models import *
from utils import get_platform_path
import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess

def get_parameters(model, bias=False):
    import torch.nn as nn
    modules_skipped = (
        nn.ReLU,
        nn.MaxPool2d,
        nn.Dropout2d,
        nn.Sequential,
        FCN32s,
        FCN16s,
        FCN8s,
    )
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if bias:
                yield m.bias
            else:
                yield m.weight
        elif isinstance(m, nn.ConvTranspose2d):
            # weight is frozen because it is just a bilinear upsampling
            if bias:
                assert m.bias is None
        elif isinstance(m, modules_skipped):
            continue
        else:
            raise ValueError('Unexpected module: %s' % str(m))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FCN32s Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='epoch')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='whether to use cuda')
    parser.add_argument('--net', default='resnet18', help='network type')
    args = parser.parse_args()

    # detect device
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    # data/model/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir = get_platform_path()

    # datasets VOC2011 and others are subset of VOC2012
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    trainset = torchvision.datasets.VOCSegmentation(root=data_dir, download=True, year='2012',
                                                    image_set='train', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)

    valset = torchvision.datasets.VOCSegmentation(root=data_dir, download=True, year='2012',
                                                  image_set='val', transform=transform_val)
    valloader = torch.utils.data.DataLoader(valset, batch_size=16, shuffle=False, num_workers=4)

    # Model
    print('==> Building model..')
    net = FCN32s(n_class=21)
    net = net.to(device)

    best_acc = 0
    start_epoch = 0

    # Load checkpoint.
    if args.resume:
        print("==> Resuming from checkpoint..")
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('{}/{}_cifar10_model.pth'.format(checkpoint_dir, args.net))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        vgg16 = VGG16(pretrained=True)
        net.copy_params_from_vgg16(vgg16)

    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    print(net)

    # optimizer
    optim = torch.optim.SGD(
        [
            {'params': get_parameters(model, bias=False)},
            {'params': get_parameters(model, bias=True),
             'lr': args.lr * 2, 'weight_decay':  0},
        ],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    if args.resume:
        optim.load_state_dict(checkpoint['optim_state_dict'])