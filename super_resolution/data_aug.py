import argparse
from utils import *

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch super resolution data augmentation')

    # file-setting
    parser.add_argument('--input', type=str, default='input', help='directory of input data for augmentation')
    parser.add_argument('--output', type=str, default='output', help='directory of output data for augmentation')

    # configuration
    parser.add_argument('--upscaleFactor', '-uf', type=int, default=3, help='super resolution upscale factor')
    parser.add_argument('--size', type=int, default=33, help='size of crop image')
    parser.add_argument('--stride', type=int, default=14, help='stride of crop image')
    parser.add_argument('--use_bicubic', action='store_true',
                        help='whether to use Bicubic Interpolation after downsampling')
    parser.add_argument('--scale', dest='scale', nargs='+', default='1.0', help='scale for data augmentation')
    parser.add_argument('--rotation', dest='rotation', nargs='+', default='0', help='rotation for data augmentation')
    args = parser.parse_args()

    print(args)

    # data/models/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir = get_platform_path()
    data_train_dir, data_test_dir, data_val_dir = data_dir, data_dir, data_dir
