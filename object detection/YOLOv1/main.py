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
from YOLOv1_voc import VocDatasetDetection


def YOLOv1_transforms(img, target):
    print("YOLOv1_transforms")
    return img, target


def YOLOv1_collate_fn(batch_lst):
    print("YOLOv1_collate_fn")
    print(batch_lst)
    return batch_lst


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pytorch YOLOv1 Training")

    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='epoch')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='whether to use cuda')

    parser.add_argument('--net', default='resnet18', help='network type')
    parser.add_argument('--grid', default=7, type=int, help="size of grid")
    parser.add_argument('--bbnd', default=2, type=int, help="size of Bounding")
    parser.add_argument('--classes', default=20, type=int, help="size of classes, here is 20 in voc2012")
    parser.add_argument('--resize_x', default=224, type=int, help="X length of the image after resizing")
    parser.add_argument('--resize_y', default=224, type=int, help="Y length of the image after resizing")

    args = parser.parse_args()

    # detect device
    print("CUDA Available:", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    # data/models/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir = get_platform_path()
    transform_train = transforms.Compose([
        transforms.Resize((args.resize_x, args.resize_y)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # dataset VOC2011 and others are subset of VOC2012
    trainset = VocDatasetDetection(root=data_dir, download=True, year='2012',
                                   image_set='train', transform=transform_train,
                                   grid=args.grid, bbnd=args.bbnd, classes=args.classes,
                                   resize=(args.resize_x, args.resize_y))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

    # valset = torchvision.dataset.VOCDetection(root=data_dir, download=True, year='2012',
    #                                            image_set='val', transforms=YOLOv1_transforms)
    # valloader = torch.utils.data.DataLoader(valset, batch_size=10, shuffle=False,
    #                                         num_workers=2, collate_fn=voc_collate_fn)

    # for iter_num, (img_batch, box_batch, w_batch, h_batch, img_id_list) in enumerate(trainloader):
    #     print("loader")
    #     print(iter_num, img_batch, box_batch.shape, w_batch.shape, h_batch.shape, len(img_id_list))
    #     break

    i = 0
    for iter, (img, target) in enumerate(trainloader):
        print(img, target)
        break
