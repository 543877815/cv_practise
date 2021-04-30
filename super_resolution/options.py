import argparse

parser = argparse.ArgumentParser(description='Pytorch super resolution example')

# global configuration
parser.add_argument('--configs', '-c', dest="filename", metavar='FILE', help='path to the configs file',
                    default='configs/srcnn.yaml')
parser.add_argument('--config_priority', default='yaml', choices=('args', 'yaml'),
                    help='optimizer to use (args | yaml )')
parser.add_argument('--use_relative', default=False, action="store_true",
                    help='whether to use relative path to for data/model/checkpoint/log')
parser.add_argument('--preprocess', default=False, action="store_true", help='whether to use data preprocessing')

# cuda && Hardware configuration
parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda')
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')
parser.add_argument('--cpu', action='store_true', help='use cpu only')
parser.add_argument('--n_GPUs', type=int, default=1, help='number of GPUs')

# hyper-parameters
parser.add_argument('--training_batch_size', type=int, default=16, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train for')
parser.add_argument('--seed', type=int, default=123, help='random seed to use')

# models configuration
parser.add_argument('--upscaleFactor', '-uf', type=int, default=3, help='super resolution upscale factor')
parser.add_argument('--model', '-m', type=str, default='srcnn', help='models that going to use')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')

# data configuration
parser.add_argument('--dataset', type=str, default='urban100', help='data that going to use')
parser.add_argument('--single_channel', action='store_true', help='whether to use specific channel')
parser.add_argument('--num_workers', type=int, default=1, help='number of worker for data loader')
parser.add_argument('--use_bicubic', type=bool, default=True, help='where to use bicubic to resize LR to HR')
parser.add_argument('--use_h5py', action='store_true', help='whether to use .h5 file as data input')
parser.add_argument('--h5py_input', type=str, default='h5py_input', help='.h5 file data for training')
parser.add_argument('--train_LR_dir', type=str, default='train_LR_dir', help='low resolution data for training')
parser.add_argument('--train_HR_dir', type=str, default='train_HR_dir', help='high resolution data for training')
parser.add_argument('--test_LR_dir', type=str, default='val_LR_dir', help='low resolution data for validation')
parser.add_argument('--test_HR_dir', type=str, default='val_HR_dir', help='high resolution data for validation')

parser.add_argument('--color', type=str, default='RGB', help='color space to use, RGB/YCbCr')

# Optimization specifications
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--decay', type=str, default='200', help='learning rate decay type')
parser.add_argument('--gamma', type=float, default=0.5, help='learning rate decay factor for step decay')
parser.add_argument('--optimizer', default='ADAM', choices=('SGD', 'ADAM', 'RMSprop'),
                    help='optimizer to use (SGD | ADAM | RMSprop)')
parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='ADAM beta')
parser.add_argument('--epsilon', type=float, default=1e-8, help='ADAM epsilon for numerical stability')
parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
parser.add_argument('--gclip', type=float, default=0, help='gradient clipping threshold (0 = no clipping)')

args = parser.parse_args()
