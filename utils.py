import hashlib
import os
import platform

import cv2
import math
import numpy as np
import gdown
import sys
import time
import logging

TOTAL_BAR_LENGTH = 80
LAST_T = time.time()
BEGIN_T = LAST_T


def get_platform_path():
    system = platform.system()
    data_dir, model_dir, checkpoint_dir, log_dir, dirs = '', '', '', '', []
    if system == 'Windows':
        drive, common_dir = 'F', 'cache'
        data_dir = '{}:/{}/data'.format(drive, common_dir)
        model_dir = '{}:/{}/model'.format(drive, common_dir)
        checkpoint_dir = '{}:/{}/checkpoint'.format(drive, common_dir)
        log_dir = '{}:/{}/log'.format(drive, common_dir)
        dirs = [data_dir, model_dir, checkpoint_dir, log_dir]

    elif system == 'Linux':
        common_dir = '/data'
        data_dir = '{}/data'.format(common_dir)
        model_dir = '{}/model'.format(common_dir)
        checkpoint_dir = '{}/checkpoint'.format(common_dir)
        log_dir = '{}/log'.format(common_dir)
        dirs = [data_dir, model_dir, checkpoint_dir, log_dir]

    for dir in dirs:
        if not os.path.exists(dir):
            os.mkdir(dir)

    return data_dir, model_dir, checkpoint_dir, log_dir


def _md5sum(filename, blocksize=65536):
    hash = hashlib.md5()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(blocksize), b''):
            hash.update(block)
    return hash.hexdigest()


def cached_download(url, path, md5=None, quiet=False, postprocess=None):
    def check_md5(path, md5):
        print('[{:s}] Checking md5 ({:s})'.format(path, md5))
        return _md5sum(path) == md5

    if os.path.exists(path) and not md5:
        print('[{:s}] File exists ({:s})'.format(path, _md5sum(path)))
    elif os.path.exists(path) and md5 and check_md5(path, md5):
        pass
    else:
        dirpath = os.path.dirname(path)
        if not os.path.exists(dirpath):
            os.makedirs(dirpath)
        gdown.download(url, path, quiet=quiet)

    if postprocess is not None:
        postprocess(path)

    return path


def progress_bar(current, total, msg=None):
    global LAST_T, BEGIN_T
    if current == 0:
        BEGIN_T = time.time()  # Reset for new bar.

    current_len = int(TOTAL_BAR_LENGTH * (current + 1) / total)
    rest_len = int(TOTAL_BAR_LENGTH - current_len) - 1

    sys.stdout.write(' %d/%d' % (current + 1, total))
    sys.stdout.write(' [')
    for i in range(current_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    current_time = time.time()
    step_time = current_time - LAST_T
    LAST_T = current_time
    total_time = current_time - BEGIN_T

    time_used = '  Step: %s' % format_time(step_time)
    time_used += ' | Tot: %s' % format_time(total_time)
    if msg:
        time_used += ' | ' + msg

    msg = time_used
    sys.stdout.write(msg)

    if current < total - 1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()


# return the formatted time
def format_time(seconds):
    days = int(seconds / 3600 / 24)
    seconds = seconds - days * 3600 * 24
    hours = int(seconds / 3600)
    seconds = seconds - hours * 3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    seconds_final = int(seconds)
    seconds = seconds - seconds_final
    millis = int(seconds * 1000)

    output = ''
    time_index = 1
    if days > 0:
        output += str(days) + 'D'
        time_index += 1
    if hours > 0 and time_index <= 2:
        output += str(hours) + 'h'
        time_index += 1
    if minutes > 0 and time_index <= 2:
        output += str(minutes) + 'm'
        time_index += 1
    if seconds_final > 0 and time_index <= 2:
        output += str(seconds_final) + 's'
        time_index += 1
    if millis > 0 and time_index <= 2:
        output += str(millis) + 'ms'
        time_index += 1
    if output == '':
        output = '0ms'
    return output


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpeg', '.jpg', '.bmp', '.JPEG'])


# / 255
OrigT = np.array(
    [[65.481, 128.553, 24.966],
     [-37.797, -74.203, 112.0],
     [112.0, -93.786, -18.214]])

OrigOffset = np.array([16, 128, 128])

# OrigT_inv = np.array([[0.00456621,  0.,  0.00625893],
#           [0.00456621, -0.00153632, -0.00318811],
#           [0.00456621,  0.00791071,  0.]])
OrigT_inv = np.linalg.inv(OrigT)


def rgb2ycbcr(rgb_img):
    if rgb_img.shape[2] == 1:
        return rgb_img
    if rgb_img.dtype == float:
        T = 1.0 / 255.0
        offset = 1 / 255.0
    elif rgb_img.dtype == np.uint8:
        T = 1.0 / 255.0
        offset = 1.0
    elif rgb_img.dtype == np.uint16:
        T = 257.0 / 65535.0
        offset = 257.0
    else:
        raise Exception('the dtype of image does not support')
    T = T * OrigT
    offset = offset * OrigOffset
    ycbcr_img = np.zeros(rgb_img.shape, dtype=float)
    for p in range(rgb_img.shape[2]):
        ycbcr_img[:, :, p] = T[p, 0] * rgb_img[:, :, 0] + T[p, 1] * rgb_img[:, :, 1] + T[p, 2] * rgb_img[:, :, 2] + \
                             offset[p]
    return np.array(ycbcr_img, dtype=rgb_img.dtype)


def ycbcr2rgb(ycbcr_img):
    if ycbcr_img.shape[2] == 1:
        return ycbcr_img
    if ycbcr_img.dtype == float:
        T = 255.0
        offset = 1.0
    elif ycbcr_img.dtype == np.uint8:
        T = 255.0
        offset = 255.0
    elif ycbcr_img.dtype == np.uint16:
        T = 65535.0 / 257.0
        offset = 65535.0
    else:
        raise Exception('the dtype of image does not support')
    T = T * OrigT_inv
    offset = offset * np.matmul(OrigT_inv, OrigOffset)
    rgb_img = np.zeros(ycbcr_img.shape, dtype=float)
    for p in range(rgb_img.shape[2]):
        rgb_img[:, :, p] = T[p, 0] * ycbcr_img[:, :, 0] + T[p, 1] * ycbcr_img[:, :, 1] + T[p, 2] * ycbcr_img[:, :, 2] - \
                           offset[p]
    return np.array(rgb_img.clip(0, 255), dtype=ycbcr_img.dtype)


# Here is the function for shaving the edge of image
def shave(pred, gt, shave_border):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    return pred, gt


# Here is the function for PSNR calculation
def PSNR(pred, gt):
    imdff = pred - gt
    rmse = np.mean(np.mean(imdff ** 2, axis=1), axis=0)
    if rmse == 0:
        return 100
    return 10 * math.log10(255 * 255 / rmse)


# Here is the function for SSIM calculation
def SSIM_index(pred, gt):
    height, width = pred.shape[:2]
    kernel_size = 11
    padding = kernel_size // 2
    std = 1.5
    if height < kernel_size or width < kernel_size:
        return float('-inf')
    K = [0.01, 0.03]
    L = 255
    C1, C2 = (K[0] * L) ** 2, (K[1] * L) ** 2
    pred, gt = np.array(pred, dtype=float), np.array(gt, dtype=float)
    mu1 = cv2.GaussianBlur(pred, (kernel_size, kernel_size), std, borderType=None)
    mu2 = cv2.GaussianBlur(gt, (kernel_size, kernel_size), std, borderType=None)
    mu1, mu2 = shave(mu1, mu2, padding)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(pred * pred, (kernel_size, kernel_size), std, borderType=None)
    sigma2_sq = cv2.GaussianBlur(gt * gt, (kernel_size, kernel_size), std, borderType=None)
    sigma1_sq, sigma2_sq = shave(sigma1_sq, sigma2_sq, padding)
    sigma1_sq, sigma2_sq = sigma1_sq - mu1_sq, sigma2_sq - mu2_sq
    sigma12 = cv2.GaussianBlur(pred * gt, (kernel_size, kernel_size), std, borderType=None)
    sigma12, _ = shave(sigma12, sigma12, padding)
    sigma12 = sigma12 - mu1_mu2
    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    else:
        raise NotImplementedError()
    mssim = np.mean(ssim_map)
    return mssim, ssim_map

# TODO: IFC
def IFC(pred, gt):
    pass

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "a")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
