import shutil
import tarfile
import random
import h5py
import os
from utils import get_platform_path, is_image_file, rgb2ycbcr
from six.moves import urllib
import torch.utils.data as data
from PIL import Image
import numpy as np
from tqdm import tqdm
import torch





class DatasetFromTwoFolder(data.Dataset):
    def __init__(self, LR_dir, HR_dir, train=False, config=None, transform=None, target_transform=None):
        super(DatasetFromTwoFolder, self).__init__()

        LR_filenames = os.listdir(LR_dir)
        HR_filenames = os.listdir(HR_dir)
        LR_filenames.sort(key=lambda x: x[:-4])
        HR_filenames.sort(key=lambda x: x[:-4])

        self.LR_image_filenames = [os.path.join(LR_dir, x) for x in LR_filenames if is_image_file(x)]
        self.HR_image_filenames = [os.path.join(HR_dir, x) for x in HR_filenames if is_image_file(x)]
        self.config = config
        self.train = train

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        img = self.load_img(self.LR_image_filenames[item])
        target = self.load_img(self.HR_image_filenames[item])
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        if self.train:
            img, target = self.get_patch(LR_img=img, HR_img=target)
        return img, target

    def __len__(self):
        return len(self.LR_image_filenames)

    def get_patch(self, LR_img, HR_img):
        height, width = HR_img.shape[1], HR_img.shape[2]
        size = self.config.img_size

        if self.config.use_bicubic:
            tp = size
            ip = size
        else:
            tp = size
            ip = tp // self.config.upscaleFactor

        ix = random.randrange(0, width - ip + 1)
        iy = random.randrange(0, height - ip + 1)

        if self.config.use_bicubic:
            tx, ty = ix, iy
        else:
            tx, ty = self.config.upscaleFactor * ix, self.config.upscaleFactor * iy

        return LR_img[:, iy:iy + ip, ix:ix + ip], HR_img[:, ty: ty + tp, tx:tx + tp]

    def load_img(self, filepath):
        img = Image.open(filepath)
        if len(img.split()) == 1:
            return img
        img = img.convert('RGB')
        if self.config.color_space == 'RGB':
            return img
        elif self.config.color_space == 'YCbCr':
            img_ycrcb = rgb2ycbcr(np.array(img, dtype=np.uint8))
            if self.config.num_channels == 1:
                return Image.fromarray(img_ycrcb[:, :, 0])
            else:
                return Image.fromarray(img_ycrcb)
        else:
            raise Exception("the color space does not exist")


class DatasetForSRFromFolder(data.Dataset):
    def __init__(self, image_dir, config, transform=None):
        super(DatasetForSRFromFolder, self).__init__()
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
            if self.config.num_channels == 1:
                return img_ycrcb[:, :, 0]
            else:
                return img_ycrcb
        else:
            raise Exception("the color space does not exist")


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
                img = np.array(f['lr'][idx], dtype=np.uint8)
                target = np.array(f['hr'][idx], dtype=np.uint8)
                if self.transform:
                    img = self.transform(img)
                if self.target_transform:
                    target = self.target_transform(target)
                return img, target
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


def buildRawData(origin_HR_dir, train_HR_dir, train_LR_dir, config):
    if config.force_rebuild:
        shutil.rmtree(train_HR_dir)
        shutil.rmtree(train_LR_dir)
    if os.path.exists(train_HR_dir) and len(os.listdir(train_HR_dir)) > 0 and \
            os.path.exists(train_LR_dir) and len(os.listdir(train_LR_dir)) > 0:
        assert len(os.listdir(train_HR_dir)) == len(os.listdir(train_LR_dir)), \
         "The number of train data is not equal to the number of test data, please rebuild the data."
        return
    if not os.path.exists(train_LR_dir):
        os.mkdir(train_LR_dir)
    if not os.path.exists(train_HR_dir):
        os.mkdir(train_HR_dir)
    for image in tqdm(os.listdir(origin_HR_dir)):
        abs_image = os.path.join(origin_HR_dir, image)
        img_HR = Image.open(abs_image).convert("RGB")
        size = img_HR.size
        for scale in config.scales:
            scale_x, scale_y = int(size[0] * scale), int(size[1] * scale)
            img_HR_scale = img_HR.resize((scale_x, scale_y), Image.BICUBIC)
            for upscaleFactor in config.upscaleFactor:
                scale_x = scale_x - (scale_x % upscaleFactor)
                scale_y = scale_y - (scale_y % upscaleFactor)
                img_HR_scale = img_HR_scale.resize((scale_x, scale_y), Image.BICUBIC)
                img_LR = img_HR_scale.resize((scale_x // upscaleFactor, scale_y // upscaleFactor), Image.BICUBIC)
                if config.use_bicubic:
                    img_LR = img_LR.resize((scale_x, scale_y), Image.BICUBIC)
                # avoid size too low
                if img_LR.size[0] < config.img_size:
                    continue
                for rotation in config.rotations:
                    img_LR = img_LR.rotate(rotation, expand=True)
                    img_HR_scale = img_HR_scale.rotate(rotation, expand=True)

                    for flip in config.flips:
                        if flip == 1:
                            img_LR = img_LR.transpose(Image.FLIP_LEFT_RIGHT)
                            img_HR_scale = img_HR_scale.transpose(Image.FLIP_LEFT_RIGHT)
                        elif flip == 2:
                            img_LR = img_LR.transpose(Image.FLIP_TOP_BOTTOM)
                            img_HR_scale = img_HR_scale.transpose(Image.FLIP_TOP_BOTTOM)
                        elif flip == 3:
                            img_LR = img_LR.transpose(Image.FLIP_TOP_BOTTOM)
                            img_LR = img_LR.transpose(Image.FLIP_LEFT_RIGHT)
                            img_HR_scale = img_HR_scale.transpose(Image.FLIP_TOP_BOTTOM)
                            img_HR_scale = img_HR_scale.transpose(Image.FLIP_LEFT_RIGHT)

                        def get_suffix(image):
                            for extension in ['.png', '.jpeg', '.jpg', '.bmp', '.JPEG']:
                                if image.endswith(extension):
                                    return extension

                        suffix = get_suffix(image)
                        image_name = image.replace(suffix,
                                                   "_x{}_s{}_r{}_f{}{}".format(upscaleFactor, scale,
                                                                               rotation, flip, suffix))
                        path_LR = os.path.join(train_LR_dir, image_name)
                        path_HR = os.path.join(train_HR_dir, image_name)
                        img_LR.save(path_LR)
                        img_HR_scale.save(path_HR)


def BSD300(config):
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

    origin_HR_dir = data_dir + '/train'
    train_LR_dir = data_dir + '/LR_x{}'.format(config.upscaleFactor)
    train_HR_dir = data_dir + '/HR_x{}'.format(config.upscaleFactor)

    print("===> Generate low resolution images:")
    buildRawData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
    return train_LR_dir, train_HR_dir


def BSDS500(config):
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

    origin_HR_dir = data_dir + '/train'
    train_LR_dir = data_dir + '/LR_x{}'.format(config.upscaleFactor)
    train_HR_dir = data_dir + '/HR_x{}'.format(config.upscaleFactor)

    print("===> Generate low resolution images:")
    buildRawData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
    return train_LR_dir, train_HR_dir


def images91(config):
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "91-image")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/91-images"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    origin_HR_dir = data_dir + '/HR'
    train_LR_dir = data_dir + '/LR_x{}'.format(config.upscaleFactor)
    train_HR_dir = data_dir + '/HR_x{}'.format(config.upscaleFactor)

    print("===> Generate low resolution images:")
    buildRawData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
    return train_LR_dir, train_HR_dir


def Set5(config):
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "Set5")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/Set5"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    origin_HR_dir = data_dir + '/HR'
    train_LR_dir = data_dir + '/LR_x{}'.format(config.upscaleFactor)
    train_HR_dir = data_dir + '/HR_x{}'.format(config.upscaleFactor)

    print("===> Generate low resolution images:")
    buildRawData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
    return train_LR_dir, train_HR_dir


def Set14(config):
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "Set14")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/Set14"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    origin_HR_dir = data_dir + '/HR'
    train_LR_dir = data_dir + '/LR_x{}'.format(config.upscaleFactor)
    train_HR_dir = data_dir + '/HR_x{}'.format(config.upscaleFactor)

    print("===> Generate low resolution images:")
    buildRawData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
    return train_LR_dir, train_HR_dir


def B100(config):
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "B100")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/B100"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    origin_HR_dir = data_dir + '/HR'
    train_LR_dir = data_dir + '/LR_x{}'.format(config.upscaleFactor)
    train_HR_dir = data_dir + '/HR_x{}'.format(config.upscaleFactor)

    print("===> Generate low resolution images:")
    buildRawData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
    return train_LR_dir, train_HR_dir


def Urban100(config):
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "Urban100")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/Urban100"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    origin_HR_dir = data_dir + '/HR'
    train_LR_dir = data_dir + '/LR_x{}'.format(config.upscaleFactor)
    train_HR_dir = data_dir + '/HR_x{}'.format(config.upscaleFactor)

    print("===> Generate low resolution images:")
    buildRawData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
    return train_LR_dir, train_HR_dir


def Manga109(config):
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "Manga109")

    if not os.path.exists(data_dir):
        url = "https://github.com/502408764/Manga109"
        print("===> Downloading url:", url)
        os.system('git clone {} {}'.format(url, data_dir))

    origin_HR_dir = data_dir + '/HR'
    train_LR_dir = data_dir + '/LR_x{}'.format(config.upscaleFactor)
    train_HR_dir = data_dir + '/HR_x{}'.format(config.upscaleFactor)

    print("===> Generate low resolution images:")
    buildRawData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
    return train_LR_dir, train_HR_dir


def DIV2K():
    # data/models/checkpoint in different platform
    data_dir, _, _, _ = get_platform_path()
    data_dir = os.path.join(data_dir, "DIV2K")
    if not os.path.exists(data_dir):
        url = "https://cv.snu.ac.kr/research/EDSR/DIV2K.tar"
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
