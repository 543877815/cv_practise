"""Train CIFAR10 with PyTorch"""

"""Fork from https://github.com/kuangliu/pytorch-cifar"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import argparse
import platform
import os

from tqdm import tqdm
from models import *

if __name__ == "__main__":
    # parse argument
    parser = argparse.ArgumentParser(description='PyTorch CIFAR Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='epoch')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--use_cuda', action='store_true', help='whether to use cuda')
    args = parser.parse_args()

    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
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
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root=dataDir, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=dataDir, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=25, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model
    print('==> Building model..')
    net = LeNet()
    # net = VGG('VGG19')
    # net = ResNet101()
    # net = PreActResNet18()
    # net = GoogLeNet()
    # net = DenseNet121()
    # net = ResNeXt29_2x64d()
    # net = MobileNet()
    # net = MobileNetV2()
    # net = DPN92()
    # net = ShuffleNetG2()
    # net = SENet18()
    # net = ShuffleNetV2(1)
    # net = EfficientNetB0()

    net = net.to(device)
    if device == 'cuda':
        net = torch.nn.DataParallel(net, device_ids=[0])
        cudnn.benchmark = True

    best_acc = 0
    start_epoch = 0

    # Load checkpoint.
    if args.resume:
        print("==> Resuming from checkpoint..")
        assert os.path.isdir(checkpointDir), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('{}/LeNet_cifar10_model.pth'.format(checkpointDir))
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)


    # progressbar
    def pbar_desc(progressbar, loader_length, loss, batch_idx, correct, total):
        bar = "=" * round((batch_idx / loader_length) * 40) + ">"
        progressbar.set_description('[%-40s] Loss: %.3f | Acc: %.3f%% (%d/%d)'
                                    % (bar, loss / (batch_idx + 1), 100. * correct / total, correct, total))


    # Training
    def train(epoch):
        print('\nEpoch: {}'.format(epoch))
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        progressbar = tqdm(enumerate(trainloader))

        for batch_idx, (inputs, targets) in progressbar:
            inputs, targets, = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            pbar_desc(progressbar, len(trainloader), train_loss, batch_idx, correct, total)


    # Testing
    def test(epoch):
        global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        progressbar = tqdm(enumerate(testloader))

        with torch.no_grad():
            for batch_idx, (inputs, targets) in progressbar:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                test_loss += loss.item()

                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                pbar_desc(progressbar, len(testloader), test_loss, batch_idx, correct, total)

        # save checkpoint
        acc = 100 * correct / total
        if acc > best_acc:
            print('Saving..')
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            torch.save(state, "{}/LeNet_cifar10_model.pth".format(checkpointDir))
            best_acc = acc


    for epoch in range(start_epoch, start_epoch + args.epoch):
        train(epoch)
        test(epoch)
