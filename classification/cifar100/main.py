"""Train CIFAR100 with PyTorch"""
"""Implemente refer to  https://github.com/kuangliu/pytorch-cifar && https://github.com/weiaicunzai/pytorch-cifar100"""

import os
import platform
import sys
import argparse
import time
from datetime import datetime
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import *
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch CIFAR100 Training')

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

    trainset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)

    testset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)

    # model
    net = get_network(args)
    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    best_acc = 0
    start_epoch = 0
    acc_top1 = 0
    acc_top5 = 0
    # Load checkpoint.
    if args.resume:
        print("==> Resuming from checkpoint..")
        assert os.path.isdir(checkpoint_dir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('{}/{}_cifar100_model.pth'.format(checkpoint_dir, args.net))
        net.load_state_dict(checkpoint['net'])
        acc_top1 = checkpoint['acc_top1']
        acc_top5 = checkpoint['acc_top5']
        start_epoch = checkpoint['epoch']
        best_acc = acc_top5

    # loss function, optimizer and learning rate decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)

    # progressbar
    def pbar_desc(progressbar, loader_length, loss, batch_idx, correct, total):
        correct_top1 = correct[0]
        correct_top5 = correct[1]
        bar = "=" * round((batch_idx / loader_length) * 40) + ">"
        progressbar.set_description('[%-40s] Loss: %.3f | Top1_Acc: %.3f%% (%d/%d) | Top5_Acc: %.3f%% (%d/%d)'
                                    % (bar, loss / (batch_idx + 1),
                                       100. * correct_top1 / total, correct_top1, total,
                                       100. * correct_top5 / total, correct_top5, total))

    # Training
    def train(epoch):
        print('\nEpoch: {}'.format(epoch))
        net.train()
        train_loss = 0
        correct_top1 = 0.0
        correct_top5 = 0.0
        total = 0
        progressbar = tqdm(enumerate(trainloader))

        for batch_idx, (inputs, targets) in progressbar:
            inputs, targets, = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_scheduler.step()

            train_loss += loss.item()
            _, predicted = outputs.topk(5, 1, largest=True, sorted=True)

            total += targets.size(0)

            targets = targets.view(targets.size(0), -1).expand_as(predicted)
            correct = predicted.eq(targets).float()
            correct_top1 += correct[:, :1].sum()
            correct_top5 += correct[:, :5].sum()

            pbar_desc(progressbar, len(trainloader), train_loss, batch_idx, [correct_top1, correct_top5], total)


    # Testing
    def test(epoch):
        global best_acc, acc_top5, acc_top1
        net.eval()
        test_loss = 0
        correct_top1 = 0.0
        correct_top5 = 0.0
        total = 0
        progressbar = tqdm(enumerate(testloader))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in progressbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = outputs.topk(5, 1, largest=True, sorted=True)
                total += targets.size(0)

                targets = targets.view(targets.size(0), -1).expand_as(predicted)
                correct = predicted.eq(targets).float()
                correct_top1 += correct[:, :1].sum()
                correct_top5 += correct[:, :5].sum()

                pbar_desc(progressbar, len(testloader), test_loss, batch_idx, [correct_top1, correct_top5], total)

        # save checkpoint
        acc_top1 = 100 * correct_top1 / total
        acc_top5 = 100 * correct_top5 / total
        if acc_top5 > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc_top1': acc_top1,
                'acc_top5': acc_top5,
                'epoch': epoch,
            }
            torch.save(state, "{}/{}_cifar100_model.pth".format(checkpoint_dir, args.net))
            best_acc = acc_top5


    for epoch in range(start_epoch, start_epoch + args.epoch):
        train(epoch)
        test(epoch)
