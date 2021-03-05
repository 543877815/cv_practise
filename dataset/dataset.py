import tarfile

import h5py
import os
from utils import get_platform_path, is_image_file, get_logger, rgb2ycbcr
from six.moves import urllib
import torch.utils.data as data
from PIL import Image
import numpy as np


class DatasetFromOneFolder(data.Dataset):
    def __init__(self, image_dir, config=None, transform=None, target_transform=None):
        super(DatasetFromOneFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
        self.config = config

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        img = self.load_img(self.image_filenames[item])
        target = img.copy()
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.image_filenames)

    def load_img(self, filepath):
        if self.config.color == 'YCbCr':
            img = Image.open(filepath).convert('YCbCr')
        elif self.config.color == 'RGB':
            img = Image.open(filepath).convert('RGB')
        else:
            raise Exception("the color space does not exist")
        y, Cb, Cr = img.split()
        if self.config.single_channel:
            return y
        else:
            return img


class DatasetFromTwoFolder(data.Dataset):
    def __init__(self, LR_dir, HR_dir, config=None, transform=None, target_transform=None):
        super(DatasetFromTwoFolder, self).__init__()

        LR_filenames = os.listdir(LR_dir)
        LR_filenames.sort(key=lambda x: x[:-4])
        HR_filenames = os.listdir(HR_dir)
        HR_filenames.sort(key=lambda x: x[:-4])

        self.LR_image_filenames = [os.path.join(LR_dir, x) for x in LR_filenames if is_image_file(x)]
        self.HR_image_filenames = [os.path.join(HR_dir, x) for x in HR_filenames if is_image_file(x)]
        self.config = config

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        img = self.load_img(self.LR_image_filenames[item])
        target = self.load_img(self.HR_image_filenames[item])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        return img, target

    def __len__(self):
        return len(self.LR_image_filenames)

    def load_img(self, filepath):
        img = Image.open(filepath)
        if len(img.split()) == 1:
            return img
        else:
            img.convert('RGB')

        if self.config.color == 'RGB':
            return img
        elif self.config.color == 'YCbCr':
            img_ycrcb = rgb2ycbcr(np.array(img, dtype=np.uint8))
            if self.config.single_channel:
                return img_ycrcb[:, :, 0]
            else:
                return img_ycrcb
        else:
            raise Exception("the color space does not exist")


class DataSuperResolutionFromFolder(data.Dataset):
    def __init__(self, image_dir, config, transform=None):
        super(DataSuperResolutionFromFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if is_image_file(x)]
        self.config = config
        self.transform = transform

    def __getitem__(self, item):
        image_filename = self.image_filenames[item]
        img = self.load_img(image_filename)
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(image_filename)

    def load_img(self, filepath):
        img = Image.open(filepath)
        if len(img.split()) == 1:
            return img
        else:
            img.convert('RGB')

        if self.config.color == 'RGB':
            return img
        elif self.config.color == 'YCbCr':
            img_ycrcb = rgb2ycbcr(np.array(img, dtype=np.uint8))
            if self.config.single_channel:
                return img_ycrcb[:, :, 0]
            else:
                return img_ycrcb
        else:
            raise Exception("the color space does not exist")


class TrainDataset(data.Dataset):
    def __init__(self, h5_file, transform=None, target_transform=None):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file
        self.transform = transform
        self.target_transform = target_transform


    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            img = np.array(f['lr'][idx], dtype=np.uint8)
            target = np.array(f['hr'][idx], dtype=np.uint8)
            if self.transform:
                img = self.transform(img)
            if self.target_transform:
                target = self.target_transform(target)
            return img, target

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


class EvalDataset(data.Dataset):
    def __init__(self, h5_file):
        super(EvalDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][str(idx)][:, :] / 255., 0), np.expand_dims(f['hr'][str(idx)][:, :] / 255., 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


def BSD300():
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "BSDS300/images")

    if not os.path.exists(data_dir):
        url = "http://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz"
        print("===> Downloading url:", url)

        data = urllib.request.urlopen(url)
        file_path = os.path.join(data_dir, os.path.basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("===> Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, data_dir)

        os.remove(file_path)

    return data_dir


def BSDS500():
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "BSR_bsds500/BSR/BSDS500/data/images")
    if not os.path.exists(data_dir):
        url = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
        print("===> Downloading url:", url)

        data = urllib.request.urlopen(url)
        file_path = os.path.join(data_dir, os.path.basename(url))
        with open(file_path, 'wb') as f:
            f.write(data.read())

        print("===> Extracting data")
        with tarfile.open(file_path) as tar:
            for item in tar:
                tar.extract(item, data_dir)

        os.remove(file_path)


def images91():
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "91-image")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/91-images"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    return data_dir


def Set5():
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "Set5")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/Set5"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    return data_dir


def Set14():
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "Set14")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/Set14"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    return data_dir


def B100():
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "B100")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/B100"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    return data_dir


def Urban100():
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "Urban100")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/Urban100"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    return data_dir


def Manga109():
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "Manga109")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/Manga109"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    return data_dir
