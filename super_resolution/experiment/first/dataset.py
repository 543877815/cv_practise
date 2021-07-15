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


class DatasetFromTwoFolder(data.Dataset):
    """
        Using this function, we must assure the size of HR are divide even by the size of LR, respectively.
    """

    def __init__(self, LR_dir: str, HR_dir: str, train=False, config=None, transform=None, target_transform=None):
        super(DatasetFromTwoFolder, self).__init__()

        LR_filenames = os.listdir(LR_dir)
        HR_filenames = os.listdir(HR_dir)
        LR_filenames.sort(key=lambda x: x[:-4])
        HR_filenames.sort(key=lambda x: x[:-4])

        LR_filenames = LR_filenames[:config.data_range]
        HR_filenames = HR_filenames[:config.data_range]

        self.LR_image_filenames = [os.path.join(LR_dir, x) for x in LR_filenames if is_image_file(x)]
        self.HR_image_filenames = [os.path.join(HR_dir, x) for x in HR_filenames if is_image_file(x)]
        self.config = config
        self.train = train
        self.repeat = config.repeat or 1
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, item):
        img = self.load_img(self.LR_image_filenames[item // self.repeat])
        target = self.load_img(self.HR_image_filenames[item // self.repeat])
        img, target = self.augment(LR_img=img, HR_img=target)
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)
        if self.train:
            img, target = self.get_patch(LR_img=img, HR_img=target)
        return img, target

    def __len__(self):
        if self.train:
            return len(self.LR_image_filenames) * self.repeat
        else:
            return len(self.LR_image_filenames)

    # TODO test this function
    def augment(self, LR_img: PngImageFile, HR_img: PngImageFile) -> [PngImageFile, PngImageFile]:
        # flip
        flip_index = random.randint(0, len(self.config.flips) - 1)
        flip_type = self.config.flips[flip_index]
        if flip_type == 1:
            LR_img = LR_img.transpose(Image.FLIP_LEFT_RIGHT)
            HR_img = HR_img.transpose(Image.FLIP_LEFT_RIGHT)
        elif flip_type == 2:
            LR_img = LR_img.transpose(Image.FLIP_TOP_BOTTOM)
            HR_img = HR_img.transpose(Image.FLIP_TOP_BOTTOM)
        elif flip_index == 3:
            LR_img = LR_img.transpose(Image.FLIP_LEFT_RIGHT)
            HR_img = HR_img.transpose(Image.FLIP_LEFT_RIGHT)
            LR_img = LR_img.transpose(Image.FLIP_TOP_BOTTOM)
            HR_img = HR_img.transpose(Image.FLIP_TOP_BOTTOM)
        # rotation
        rotation_index = random.randint(0, len(self.config.rotations) - 1)
        angle = self.config.rotations[rotation_index]
        LR_img = LR_img.rotate(angle, expand=True)
        HR_img = HR_img.rotate(angle, expand=True)
        return LR_img, HR_img

    def get_patch(self, LR_img: torch.Tensor, HR_img: torch.Tensor) -> [torch.Tensor, torch.Tensor]:
        # TODO support multiple upscaleFactor
        scale_factor_index = random.randint(0, len(self.config.upscaleFactor) - 1)
        scale_factor_type = self.config.upscaleFactor[scale_factor_index]
        upscaleFactor = scale_factor_type
        width, height = LR_img.shape[1], LR_img.shape[2]
        size = self.config.img_size
        if self.config.same_size:
            tp = size
            ip = size
        else:
            tp = size
            ip = size // upscaleFactor
        ix = random.randrange(0, width - ip + 1)
        iy = random.randrange(0, height - ip + 1)
        if self.config.same_size:
            tx, ty = ix, iy
        else:
            tx, ty = upscaleFactor * ix, upscaleFactor * iy
        return LR_img[:, ix:ix + ip, iy:iy + ip], HR_img[:, tx:tx + tp, ty: ty + tp]

    def load_img(self, filepath):
        img = Image.open(filepath)
        if len(img.split()) == 1:
            return img
        img = img.convert('RGB')
        if self.config.color_space == 'RGB':
            return img
        elif self.config.color_space == 'YCbCr':
            img_ycrcb = rgb2ycbcr(np.array(img, dtype=np.uint8))
            if self.config.img_channels == 1:
                return Image.fromarray(img_ycrcb[:, :, 0])
            else:
                return Image.fromarray(img_ycrcb)
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
            if self.config.img_channels == 1:
                return img_ycrcb[:, :, 0]
            else:
                return img_ycrcb
        else:
            raise Exception("the color space does not exist")


def buildAugData(origin_HR_dir, train_HR_dir, train_LR_dir, config):
    if not config.buildAugData:
        return
    shutil.rmtree(train_HR_dir, ignore_errors=True)
    shutil.rmtree(train_LR_dir, ignore_errors=True)
    os.makedirs(train_HR_dir, exist_ok=True)
    os.makedirs(train_LR_dir, exist_ok=True)
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
                if config.same_size:
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
    buildAugData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
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
    buildAugData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
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
    buildAugData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
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
    buildAugData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
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
    buildAugData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
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
    buildAugData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
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
    buildAugData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
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
    buildAugData(train_LR_dir=train_LR_dir, train_HR_dir=train_HR_dir, origin_HR_dir=origin_HR_dir, config=config)
    return train_LR_dir, train_HR_dir


def DIV2K(config):
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
    # TODO: test
    train_LR_dir = os.path.join(data_dir, 'DIV2K_train_LR_bicubic', "X{}".format(config.upscaleFactor)),
    train_HR_dir = os.path.join(data_dir, 'DIV2K_train_HR')
    return train_LR_dir, train_HR_dir
