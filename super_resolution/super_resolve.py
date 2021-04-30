import argparse
import os

import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from utils import get_platform_path
from dataset.dataset import DatasetForSRFromFolder
from super_resolution.models.SRCNN.solver import SRCNNTester
from super_resolution.models.FSRCNN.solver import FSRCNNTester
from super_resolution.models.VDSR.solver import VDSRTester
from super_resolution.models.ESPCN.solver import ESPCNTester

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch super resolution')

    # cuda-configuration
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda')

    # data configuration
    parser.add_argument('--input', type=str, default='1.jpg', help='name of single low resolution image input')
    parser.add_argument('--output', type=str, default='output', help='output directory for high resolution image')
    parser.add_argument('--color', type=str, default='RGB', help='color space to use')
    parser.add_argument('--single_channel', action='store_true', help='whether to use specific channel')

    # model configuration
    parser.add_argument('--resume', '-r', type=bool, default=True, help='resume from checkpoint')
    parser.add_argument('--model', '-m', type=str, default='srcnn',
                        help='model checkpoint file used for super resolution')
    parser.add_argument('--upscaleFactor', '-uf', type=int, default=3, help='super resolution upscale factor')
    args = parser.parse_args()

    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    # data/models/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir, log_dir = get_platform_path()
    input_dir = data_dir + args.input
    output_dir = data_dir + args.output

    # ===========================================================
    # input image setting
    # ===========================================================
    print("==> Preparing data from {}".format(input_dir))
    img_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_set = DatasetForSRFromFolder(image_dir=input_dir, config=args, transform=img_transform)
    test_loader = DataLoader(dataset=test_set, batch_size=1, shuffle=False)

    # ===========================================================
    # model import & setting
    # ===========================================================
    if args.model.lower() == 'srcnn':
        Tester = SRCNNTester(config=args, test_loader=test_loader)
    elif args.model.lower() == 'fsrcnn':
        Tester = FSRCNNTester(config=args, test_loader=test_loader)
    elif args.model.lower() == 'vdsr':
        Tester = VDSRTester(config=args, test_loader=test_loader)
    elif args.model.lower() == 'espcn':
        Tester = ESPCNTester(config=args, test_loader=test_loader)
    else:
        raise Exception("the model does not exist")

    # ===========================================================
    # output and save image
    # ===========================================================
    Tester.run()
