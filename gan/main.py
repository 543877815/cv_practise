import argparse
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from gan.BasicGAN.BasicGAN import BasicGAN
from gan.WGAN.WGAN import WGAN
from gan.DCGAN.DCGAN import DCGAN
from utils import get_platform_path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # common configuration
    parser.add_argument("--model", type=str, default='gan', help="model to use")
    parser.add_argument("--epoch", type=int, default=200, help="number of epochs of training")
    parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
    parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
    parser.add_argument("--channels", type=int, default=1, help="number of image channels")
    parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")

    # dataset
    parser.add_argument("--dataset", type=str, default="mnist", help="dataset to use")

    parser.add_argument("--use_cuda", action="store_true", default=True, help="whether to use cuda")
    parser.add_argument("--num_workers", type=int, default=8,
                        help="number of cpu threads to use during batch generation")

    # for gan
    parser.add_argument("--beta1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
    parser.add_argument("--beta2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")

    # for wgan
    parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
    parser.add_argument("--clip_value", type=float, default=0.01, help="lower and upper clip value for disc. weights")

    args = parser.parse_args()

    img_shape = (args.channels, args.img_size, args.img_size)

    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    # data/models/checkpoint in different platform
    data_dir, model_dir, checkpoint_dir, log_dir = get_platform_path()

    # data preparing
    print("==> Preparing data..")
    if args.dataset.lower() == 'mnist':
        transform_train = transforms.Compose([
            transforms.Resize(args.img_size), transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform_train)
        dataloader = torch.utils.data.DataLoader(
            dataset=train_set,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers
        )
    else:
        raise Exception("the dataset does not exist")

    # model
    if args.model.lower() == 'gan':
        model = BasicGAN(config=args, dataloader=dataloader, img_size=img_shape)
    elif args.model.lower() == 'wgan':
        model = WGAN(config=args, dataloader=dataloader, img_size=img_shape)
    elif args.model.lower() == 'dcgan':
        model = DCGAN(config=args, dataloader=dataloader)
    else:
        raise Exception("the model does not exist")

    model.train()
