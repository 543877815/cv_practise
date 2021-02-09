import argparse
from math import floor
from PIL import Image
from tqdm import tqdm

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
    parser.add_argument('--upscaleFactor', '-uf', type=int, default=3, help='super resolution upscale factor')
    parser.add_argument('--width', type=int, default=33, help='width of crop image')
    parser.add_argument('--height', type=int, default=33, help='height of crop image')
    parser.add_argument('--stride', type=int, default=14, help='stride of crop image')
    parser.add_argument('--use_bicubic', action='store_true',
                        help='whether to use Bicubic Interpolation after downsampling')
    parser.add_argument('--scale', dest='scale', nargs='+', default='1.0', help='scale for data augmentation')
    parser.add_argument('--rotation', dest='rotation', nargs='+', default='0', help='rotation for data augmentation')
    args = parser.parse_args()

    args.scales = [float(x) for x in args.scale]
    args.rotations = [float(x) for x in args.rotation]

    print(args)

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
            img = img.resize((scale_x, scale_y))

            if args.use_bicubic:
                img_LR = img.resize((scale_x // args.upscaleFactor, scale_y // args.upscaleFactor), Image.BICUBIC)
                img_LR = img_LR.resize((scale_x, scale_y), Image.BICUBIC)
            else:
                img_LR = img.copy()
            for angle in args.rotations:
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
                        sub_img_LR = img_LR.crop((x1, y1, x2, y2))
                        # sub_img.show()

                        # save HR image
                        sub_img.save("{}/{}_{}.png".format(args.output_HR, os.path.basename(file_path), id))

                        # save LR image
                        if args.use_bicubic:
                            sub_img_LR.save("{}/{}_{}.png".format(args.output_LR, os.path.basename(file_path), id))
                        else:
                            LR_x, LR_y = sub_img_LR.size[0], sub_img_LR.size[1]
                            assert LR_x % args.upscaleFactor == 0, 'the image width is no divisible by {}'.format(
                                args.upscaleFactor)
                            assert LR_y % args.upscaleFactor == 0, 'the image height is no divisible by {}'.format(
                                args.upscaleFactor)
                            sub_img_LR = sub_img_LR.resize((LR_x // args.upscaleFactor, LR_y // args.upscaleFactor))
                            sub_img_LR.save("{}/{}_{}.png".format(args.output_LR, os.path.basename(file_path), id))
                        id = id + 1
