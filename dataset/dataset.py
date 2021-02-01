__package__ = 'dataset.dataset'

import tarfile
from utils import *
from six.moves import urllib
import torch.utils.data as data
from PIL import Image
import numpy as np


class DatasetFromFolder(data.Dataset):
    def __init__(self, image_dir, config=None, transform=None, target_transform=None):
        super(DatasetFromFolder, self).__init__()
        self.image_filenames = [os.path.join(image_dir, x) for x in os.listdir(image_dir) if self.is_image_file(x)]
        self.config = config
        assert self.config.channel in [0, 1, 2], 'channel illegal'

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
            img_split = img.split()  # Y, Cb, Cr

        elif self.config.color == 'RGB':
            img = Image.open(filepath).convert('RGB')
            img_split = img.split()  # R, G, B
        else:
            raise Exception("the color space does not exist")

        if self.config.single_channel:
            return img_split[self.config.channel]
        else:
            return img

    @staticmethod
    def is_image_file(filename):
        return any(filename.endswith(extension) for extension in ['.png', 'jpeg', 'jpg'])


def BSD300():
    # data/models/checkpoint in different platform
    data_dir, _, _ = get_platform_path()
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
