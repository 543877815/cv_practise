"""Train CIFAR10 with PyTorch"""

"""Implemente refer to  https://github.com/kuangliu/pytorch-cifar && https://github.com/weiaicunzai/pytorch-cifar100"""

import os
import platform
import sys
import argparse
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='epoch')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--use_cuda', action='store_true', help='whether to use cuda')

    args = parser.parse_args()

    # detect device
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    # data/model/checkpoint in different platform
    system = platform.system()
    dataDir, modelDir, checkpointDir, dirs = '', '', '', []

    if system == 'Windows':
        drive, commonDir = 'F', 'cache'
        dataDir = '{}:\\{}\\data'.format(drive, commonDir)
        modelDir = '{}:\\{}\\model'.format(drive, commonDir)
        checkpointDir = '{}:\\{}\\checkpoint'.format(drive, commonDir)
        dirs = [dataDir, modelDir, checkpointDir]

    elif system == 'Linux':
        commonDir = '/data'
        dataDir = '{}/data'.format(commonDir)
        modelDir = '{}/model'.format(commonDir)
        checkpointDir = '{}/checkpoint'.format(commonDir)
        dirs = [dataDir, modelDir, checkpointDir]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    # data augmentation
    print("==> Preparing data..")
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5089, 0.4874, 0.4419), (0.2683, 0.2574, 0.2771)),
    ])

    trainset = torchvision.datasets.CIFAR100(root=dataDir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root=dataDir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    # model
    net = vgg11_bn()
    # net = vgg13_bn()
    # net = vgg16_bn()
    print(net)

    # loss_function = nn.CrossEntropyLoss()
    # optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    # train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES,
    #                                                  gamma=0.2)  # learning rate decay
