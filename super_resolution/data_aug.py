import sys
import os
sys.path.insert(0, os.path.abspath('../'))

import argparse
from math import floor
from PIL import Image
from tqdm import tqdm
import numpy as np
from utils import get_platform_path, is_image_file, rgb2ycbcr
import h5py

# python data_aug.py --input /data/lifengjun/celeb-inpainting_dataset/celeba-train-256/ \
#                    --output /data/lifengjun/super-resolution_dataset/celeba_x64.h5 \
#                    --use_h5py --width 256 --height 256 --use_bicubic -uf 64 --number 27000

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch super resolution data augmentation')

    # file-setting
    parser.add_argument('--input', type=str, default='input', help='directory of input data for augmentation')
    parser.add_argument('--output_HR', type=str, default=None,
                        help='directory of high resolution output data for augmentation')
    parser.add_argument('--output_LR', type=str, default=None,
                        help='directory of low resoltuion output data for augmentation')
    parser.add_argument('--output', type=str, default='output', help='directory of h5py output data for augmentation')
    parser.add_argument('--use_h5py', action='store_true', help='whether to save as file h5py')
    parser.add_argument('--number', type=int, default=1000, required=True, help='number of data to generate')

    # configuration
    parser.add_argument('--padding', type=int, default=0, help='padding between input and label')
    parser.add_argument('--single_y', action='store_true', help='whether to extract y channel in YCrCb color space')
    parser.add_argument('--width', type=int, default=41, help='width of crop image')
    parser.add_argument('--height', type=int, default=41, help='height of crop image')
    parser.add_argument('--stride', type=int, default=41, help='stride of crop image')
    parser.add_argument('--upsampling', type=str, default='bicubic', nargs='+',
                        choices=['bicubic', 'bilinear', 'nearest', 'antialias'],
                        help='whether to use Bicubic Interpolation after degradation')
    parser.add_argument('--kernel_size', type=int, default=1, dest='kernel size for blur kernel')
    parser.add_argument('--same_size', action='store_true', help='whether the HR and LR are the same size')
    parser.add_argument('--upscaleFactor', '-uf', dest='uf', nargs='+', default='2',
                        help='super resolution upscale factor')
    parser.add_argument('--scales', dest='scales', nargs='+', default='first', help='scale for data augmentation')
    parser.add_argument('--rotations', dest='rotations', nargs='+', default='0', help='rotation for data augmentation')
    parser.add_argument('--flips', dest='flips', nargs='+', default='0', help='flip for data augmentation')
    parser.add_argument('--seed', type=int, default=1234, help='shuffle seed for np.random.shuffle')

    args = parser.parse_args()
    args.uf = [int(x) for x in args.uf]
    args.scales = [float(x) for x in args.scales]
    args.rotations = [float(x) for x in args.rotations]
    args.flips = [int(x) for x in args.flips]
    interpolation = {
        'bicubic': Image.BICUBIC,
        'bilinear': Image.BILINEAR,
        'nearest': Image.NEAREST,
        'antialias': Image.ANTIALIAS,
    }

    print(args)
    if args.output_HR is not None and not os.path.exists(args.output_HR):
        os.mkdir(args.output_HR)
    if args.output_LR is not None and not os.path.exists(args.output_LR):
        os.mkdir(args.output_LR)

    # data/models/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir, log_dir = get_platform_path()
    data_train_dir, data_test_dir, data_val_dir = data_dir, data_dir, data_dir

    image_filenames = [os.path.join(args.input, x) for x in os.listdir(args.input) if is_image_file(x)]

    image_filenames = image_filenames[: args.number]

    hr_patches = []
    lr_patches = []

    print("===> {} images".format(len(image_filenames)))

    for file_path in tqdm(image_filenames):
        id = 0
        OriImg = Image.open(file_path).convert('RGB')
        size = OriImg.size
        for uf in args.uf:
            # 放大
            for scale in args.scales:
                img = OriImg.copy()
                scale_x, scale_y = int(size[0] * scale), int(size[1] * scale)

                # ensure can be divided by arg.upscaleFactor
                scale_x = scale_x - (scale_x % uf)
                scale_y = scale_y - (scale_y % uf)
                img = img.resize((scale_x, scale_y), Image.BICUBIC)

                for upsampling in args.upsampling:
                    # 降质
                    img_LR = img.resize((scale_x // uf, scale_y // uf), interpolation[upsampling])
                    # 上采样为同一个大小
                    if args.same_size:
                        img_LR = img_LR.resize((scale_x, scale_y), interpolation[upsampling])
                    # 翻转
                    for flip in args.flips:
                        if flip == 1:
                            img_LR = img_LR.transpose(Image.FLIP_LEFT_RIGHT)
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                        elif flip == 2:
                            img_LR = img_LR.transpose(Image.FLIP_TOP_BOTTOM)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        elif flip == 3:
                            img_LR = img_LR.transpose(Image.FLIP_LEFT_RIGHT)
                            img_LR = img_LR.transpose(Image.FLIP_TOP_BOTTOM)
                            img = img.transpose(Image.FLIP_LEFT_RIGHT)
                            img = img.transpose(Image.FLIP_TOP_BOTTOM)
                        # 旋转
                        for angle in args.rotations:
                            img_LR = img_LR.rotate(angle, expand=True)
                            img = img.rotate(angle, expand=True)
                            for i in range(int(floor(scale_x / args.stride))):
                                for j in range(int(floor(scale_y / args.stride))):
                                    x1 = i * args.stride
                                    x2 = x1 + args.width
                                    y1 = j * args.stride
                                    y2 = y1 + args.height
                                    if x2 > scale_x:
                                        x2 = scale_x
                                        x1 = scale_x - args.width
                                        continue
                                    if y2 > scale_y:
                                        y2 = scale_y
                                        y1 = scale_y - args.height
                                        continue
                                    sub_img = img.crop(
                                        (x1 + args.padding, y1 + args.padding, x2 - args.padding, y2 - args.padding))

                                    # process LR image
                                    if args.same_size:
                                        sub_img_LR = img_LR.crop((x1, y1, x2, y2))
                                    else:
                                        assert x1 % uf == 0, 'the image width is no divisible by {}'.format(uf)
                                        assert x2 % uf == 0, 'the image width is no divisible by {}'.format(uf)
                                        assert y1 % uf == 0, 'the image height is no divisible by {}'.format(uf)
                                        assert y2 % uf == 0, 'the image height is no divisible by {}'.format(uf)
                                        sub_img_LR = img_LR.crop((x1 / uf, y1 / uf,
                                                                  x2 / uf, y2 / uf))
                                    if args.use_h5py:
                                        if args.single_y:
                                            sub_img = np.array(sub_img, dtype=np.uint8)
                                            sub_img_LR = np.array(np.array(sub_img_LR, dtype=np.uint8))
                                            if sub_img.shape[2] == 3:
                                                sub_img = rgb2ycbcr(sub_img)
                                                sub_img_LR = rgb2ycbcr(sub_img_LR)
                                                sub_img = sub_img[:, :, 0]
                                                sub_img_LR = sub_img_LR[:, :, 0]
                                            else:
                                                sub_img = sub_img
                                                sub_img_LR = sub_img_LR
                                        else:
                                            sub_img = np.array(sub_img).astype(np.uint8)
                                            sub_img_LR = np.array(sub_img_LR).astype(np.uint8)
                                        hr_patches.append(sub_img)
                                        lr_patches.append(sub_img_LR)
                                    else:
                                        # save HR image
                                        sub_img.save(
                                            "{}/{}_{}_x{}.png".format(args.output_HR, os.path.basename(file_path), id,
                                                                      uf))
                                        # save LR image
                                        sub_img_LR.save(
                                            "{}/{}_{}_x{}.png".format(args.output_LR, os.path.basename(file_path), id,
                                                                      uf))
                                    id = id + 1
    if args.use_h5py:
        try:
            h5_file = h5py.File(args.output, 'w')
            hr_patches = np.array(hr_patches)
            lr_patches = np.array(lr_patches)
            # shuffle using same seed
            np.random.seed(args.seed)
            np.random.shuffle(hr_patches)
            np.random.seed(args.seed)
            np.random.shuffle(lr_patches)
            print("===> {} patches".format(len(hr_patches)))

            hr = h5_file.create_dataset('hr', data=hr_patches)
            lr = h5_file.create_dataset('lr', data=lr_patches)

            for arg in vars(args):
                hr.attrs[arg] = str(getattr(args, arg))
                lr.attrs[arg] = str(getattr(args, arg))

            h5_file.close()

        except OSError as e:
            print(e)

    print("===> finished !!")
