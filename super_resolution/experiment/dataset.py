import shutil
import tarfile
import random
import h5py
import os
from utils import get_platform_path, is_image_file, rgb2ycbcr
from six.moves import urllib
import torch.utils.data as data
from PIL import Image
from PIL.PngImagePlugin import PngImageFile
import numpy as np
from tqdm import tqdm
import torch


class DatasetFromH5py(data.Dataset):
    def __init__(self, h5_file, transform=None, target_transform=None, multi_train=1):
        super(DatasetFromH5py, self).__init__()
        self.multi_train = multi_train
        self.h5_file = h5_file
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            if self.multi_train == 1:
                index = int(f['index'][idx])
                img = np.array(f['lr'][idx], dtype=np.uint8)
                target = np.array(f['hr'][idx], dtype=np.uint8)
                if self.transform:
                    img = self.transform(img)
                if self.target_transform:
                    target = self.target_transform(target)
                return index, img, target
            elif self.multi_train == 2:
                img = np.array(f['lr'][idx], dtype=np.uint8)
                target_x2 = np.array(f['hr_x2'][idx], dtype=np.uint8)
                target_x4 = np.array(f['hr_x4'][idx], dtype=np.uint8)
                if self.transform:
                    img = self.transform(img)
                if self.target_transform:
                    target_x2 = self.target_transform(target_x2)
                    target_x4 = self.target_transform(target_x4)
                return img, target_x2, target_x4
            elif self.multi_train == 3:
                img = np.array(f['lr'][idx], dtype=np.uint8)
                target_x2 = np.array(f['hr_x2'][idx], dtype=np.uint8)
                target_x4 = np.array(f['hr_x4'][idx], dtype=np.uint8)
                target_x8 = np.array(f['hr_x8'][idx], dtype=np.uint8)
                if self.transform:
                    img = self.transform(img)
                if self.target_transform:
                    target_x2 = self.target_transform(target_x2)
                    target_x4 = self.target_transform(target_x4)
                    target_x8 = self.target_transform(target_x8)
                return img, target_x2, target_x4, target_x8

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])
