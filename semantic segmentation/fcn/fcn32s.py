"""Train FCN32s with PyTorch"""
"""Implemente refer to https://github.com/wkentaro/pytorch-fcn"""

import argparse
import datetime
import os
import os.path as osp
import shlex
import subprocess

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch FCN32s Training')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--epoch', default=200, type=int, help='epoch')
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--use_cuda', action='store_true', default=True, help='whether to use cuda')
    parser.add_argument('--net', default='resnet18', help='network type')

    pass
