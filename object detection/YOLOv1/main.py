"""Train CIFAR10 with PyTorch"""

"""Fork from https://github.com/xiongzihua/pytorch-YOLO-v1"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms

import argparse
import platform
import os
from datasets import *
from tqdm import tqdm
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch YOLOv1 Training")

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
    trainset = torchvision.datasets.VOCDetection(root=data_dir, download=True, year='2012',
                                                 image_set='train')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, num_workers=2,
                                              collate_fn=voc_collate_fn)
    valset = torchvision.datasets.VOCDetection(root=data_dir, download=True, year='2012',
                                               image_set='val')
    valloader = torch.utils.data.DataLoader(valset, batch_size=10, shuffle=False,
                                            num_workers=2, collate_fn=voc_collate_fn)

    for iter_num, (img_batch, box_batch, w_batch, h_batch, img_id_list) in enumerate(trainloader):
        print(iter_num, img_batch.shape, box_batch.shape, w_batch.shape, h_batch.shape, len(img_id_list))
        break
