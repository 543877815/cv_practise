import argparse

parser = argparse.ArgumentParser()

# global configuration
parser.add_argument('--configs', '-c', dest="filename", metavar='FILE', help='path to the configs file',
                    default='configs/gan.yaml')
parser.add_argument('--config_priority', default='yaml', choices=('args', 'yaml'),
                    help='optimizer to use (args | yaml )')
parser.add_argument('--use_relative', default=False, action="store_true",
                    help='whether to use relative path to for data/models/checkpoint/log')
parser.add_argument('--preprocess', default=False, action="store_true", help='whether to use data preprocessing')

# common configuration
parser.add_argument("--model", type=str, default='gan', help="models to use")
parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image samples")

# dataset
parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use")

parser.add_argument("--use_cuda", action="store_true", default=True, help="whether to use cuda")
parser.add_argument("--num_workers", type=int, default=0,
                    help="number of cpu threads to use during batch generation")

# for gan/cgan
parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

# for wgan
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")

# for cgan
parser.add_argument("--n_classes", type=int, default=10, help="number of classes for dataset")

# for ragan
parser.add_argument("--rel_avg_gan", action="store_true", help="relativistic average GAN instead of standard")

args = parser.parse_args()
