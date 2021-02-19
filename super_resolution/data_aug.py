import argparse
from math import floor
from PIL import Image
from tqdm import tqdm
import os
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch super resolution data augmentation')

    # file-setting
    parser.add_argument('--input', type=str, default='input', help='directory of input data for augmentation')
    parser.add_argument('--output_HR', type=str, default='output_HR',
                        help='directory of high resolution output data for augmentation')
    parser.add_argument('--output_LR', type=str, default='output_LR',
                        help='directory of low resoltuion output data for augmentation')

    # configuration
    parser.add_argument('--width', type=int, default=41, help='width of crop image')
    parser.add_argument('--height', type=int, default=41, help='height of crop image')
    parser.add_argument('--stride', type=int, default=41, help='stride of crop image')
    parser.add_argument('--use_bicubic', action='store_true',
                        help='whether to use Bicubic Interpolation after downsampling')
    parser.add_argument('--upscaleFactor', '-uf', type=int, default=3, help='super resolution upscale factor')
    parser.add_argument('--scale', dest='scale', nargs='+', default='1.0', help='scale for data augmentation')
    parser.add_argument('--rotation', dest='rotation', nargs='+', default='0', help='rotation for data augmentation')
    parser.add_argument('--flip', dest='flip', nargs='+', default='0', help='flip for data augmentation')
    args = parser.parse_args()

    args.scales = [float(x) for x in args.scale]
    args.rotations = [float(x) for x in args.rotation]
    args.flips = [int(x) for x in args.flip]
    print(args)

    if not os.path.exists(args.output_HR):
        os.mkdir(args.output_HR)
    if not os.path.exists(args.output_LR):
        os.mkdir(args.output_LR)

    # data/models/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir = get_platform_path()
    data_train_dir, data_test_dir, data_val_dir = data_dir, data_dir, data_dir

    image_filenames = [os.path.join(args.input, x) for x in os.listdir(args.input) if is_image_file(x)]

    for file_path in tqdm(image_filenames):
        img = Image.open(file_path).convert('RGB')
        size = img.size
        id = 0
        for scale in args.scales:
            scale_x, scale_y = int(size[0] * scale), int(size[1] * scale)

            # ensure can be divided by arg.upscaleFactor
            scale_x = scale_x - (scale_x % args.upscaleFactor)
            scale_y = scale_y - (scale_y % args.upscaleFactor)
            img = img.resize((scale_x, scale_y), Image.BICUBIC)

            # 降质
            img_LR = img.resize((scale_x // args.upscaleFactor, scale_y // args.upscaleFactor), Image.BICUBIC)
            # 上采样为同一个大小
            if args.use_bicubic:
                img_LR = img_LR.resize((scale_x, scale_y), Image.BICUBIC)

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
                for angle in args.rotations:
                    img_LR = img_LR.rotate(angle)
                    img = img.rotate(angle)
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
                            sub_img = img.crop((x1, y1, x2, y2))

                            # sub_img.show()

                            # save HR image
                            sub_img.save("{}/{}_{}_x{}.jpg".format(args.output_HR, os.path.basename(file_path), id,
                                                                   args.upscaleFactor))

                            # save LR image
                            if args.use_bicubic:
                                sub_img_LR = img_LR.crop((x1, y1, x2, y2))
                            else:
                                assert x1 % args.upscaleFactor == 0, 'the image width is no divisible by {}'.format(
                                    args.upscaleFactor)
                                assert x2 % args.upscaleFactor == 0, 'the image height is no divisible by {}'.format(
                                    args.upscaleFactor)
                                assert y1 % args.upscaleFactor == 0, 'the image height is no divisible by {}'.format(
                                    args.upscaleFactor)
                                assert y2 % args.upscaleFactor == 0, 'the image height is no divisible by {}'.format(
                                    args.upscaleFactor)
                                sub_img_LR = img_LR.crop((x1 / args.upscaleFactor, y1 / args.upscaleFactor,
                                                          x2 / args.upscaleFactor, y2 / args.upscaleFactor))

                            sub_img_LR.save(
                                "{}/{}_{}_x{}.jpg".format(args.output_LR, os.path.basename(file_path), id,
                                                          args.upscaleFactor))
                            id = id + 1
