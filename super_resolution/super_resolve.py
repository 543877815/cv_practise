import argparse
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision.transforms import ToTensor

from dataset.dataset import BSD300
from super_resolution.models.SRCNN.solver import SRCNNTrainer
from utils import get_platform_path
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch super resolution')

    # cuda-configuration
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda')

    # data configuration
    parser.add_argument('--input', type=str, default='1.jpg', help='name of single low resolution image input')
    parser.add_argument('--dataset', type=str, default=None, help='low resolution datasets input')
    parser.add_argument('--output', type=str, default='output', help='output directory for high resolution image')

    # model configuration
    parser.add_argument('--model', type=str, default='srcnn', help='model checkpoint file used for super ressolution')
    args = parser.parse_args()

    data_dir, model_dir, checkpoint_dir = get_platform_path()
    # ===========================================================
    # input image setting
    # ===========================================================
    dataset = args.dataset
    if dataset is not None:
        if dataset == 'bsd300':
            data_dir = BSD300()
            data_test_dir = data_dir + '/test'
    else:
        pass

    img = Image.open(args.input).convert('YCbCr')
    y, cb, cr = img.split()

    # ===========================================================
    # model import & setting
    # ===========================================================
    GPU_IN_USE = torch.cuda.is_available()
    device = torch.device("cuda" if (args.use_cuda and GPU_IN_USE) else "cpu")

    model = torch.load(args.model, map_location=lambda storage, loc: storage)
    model = model.to(device)
    data = (ToTensor()(y)).view(1, -1, y.size[1], y.size[0])
    data = data.to(device)

    if GPU_IN_USE:
        cudnn.benchmark = True

    # ===========================================================
    # output and save image
    # ===========================================================
    out = model(data)
    out = out.cpu()
    out_img_y = out.data[0].numpy()
    out_img_y *= 255.0
    out_img_y = out_img_y.clip(0, 255)
    out_img_y = Image.fromarray(np.uint8(out_img_y[0]), mode='L')

    out_img_cb = cb.resize(out_img_y.size, Image.BICUBIC)
    out_img_cr = cr.resize(out_img_y.size, Image.BICUBIC)
    out_img = Image.merge('YCbCr', [out_img_y, out_img_cb, out_img_cr]).convert('RGB')

    out_img.save(args.output)
    print('output image saved to ', args.output)
