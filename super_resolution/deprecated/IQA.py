import argparse

from math import log10

import torch
import torch.backends.cudnn as cudnn
import os
from utils import get_platform_path, is_image_file
from torchvision.transforms import transforms
from PIL import Image
import numpy as np
from torchvision import utils as vutils

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch super resolution')

    # cuda-configuration
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda')

    # image-configuration
    parser.add_argument('--upscaleFactor', '-uf', type=int, default=3, help='super resolution upscale factor')
    parser.add_argument('--ground_true', type=str, default='ground_true.jpg', help='name of ground true image input')
    parser.add_argument('--output', type=str, default='output.jpg', help='output directory for high resolution image')

    # data configuration
    parser.add_argument('--color', type=str, default='RGB', help='color space to use, RGB/YCbCr')
    parser.add_argument('--num_channels', type=int, default=1, help='number of channel')

    args = parser.parse_args()

    # data/models/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir = get_platform_path()
    ground_true_dir = data_dir + args.ground_true
    output_dir = data_dir + args.output
    output_filenames = [os.path.join(output_dir, x) for x in os.listdir(output_dir) if is_image_file(x)]

    transform = transforms.ToTensor()
    criterion = torch.nn.MSELoss()

    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    for output_filename in output_filenames:
        filename = os.path.basename(output_filename)
        ground_true_filename = ground_true_dir + filename

        output_img = Image.open(output_filename).convert(args.color)
        output_img = transform(output_img)
        ground_true_img = Image.open(ground_true_filename).convert(args.color)
        ground_true_img = transform(ground_true_img)

        x, y = ground_true_img.shape[1:]
        BICUBIC_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((x // args.upscaleFactor, y // args.upscaleFactor), interpolation=Image.BICUBIC),
            transforms.Resize((x, y), interpolation=Image.BICUBIC),
            transforms.ToTensor()
        ])
        BICUBIC_img = BICUBIC_transform(ground_true_img)

        output_img, ground_true_img, BICUBIC_img, criterion = output_img.to(device), ground_true_img.to(
            device), BICUBIC_img.to(device), criterion.to(device)

        if args.num_channels == 1:
            BICUBIC_mse = criterion(BICUBIC_img[0], ground_true_img[0])
            mse = criterion(output_img[0], ground_true_img[0])
            BICUBIC_psnr = 10 * log10(1 / BICUBIC_mse.item())
            psnr = 10 * log10(1 / mse.item())
        else:
            BICUBIC_mse = criterion(BICUBIC_img, ground_true_img)
            mse = criterion(output_img, ground_true_img)
            BICUBIC_psnr = 10 * log10(1 / BICUBIC_mse.item())
            psnr = 10 * log10(1 / mse.item())
        print("{}, BICUBIC_psnr:{}".format(output_filename, BICUBIC_psnr))
        print("{}, psnr:{}".format(output_filename, psnr))
