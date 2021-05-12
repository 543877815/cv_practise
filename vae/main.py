import yaml
import argparse
import numpy as np
from attrdict import AttrDict
from utils import get_platform_path
from models import *
import torch.backends.cudnn as cudnn
from vae.models import *
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generic runner for VAE models')
    parser.add_argument('--config', '-c',
                        dest="filename",
                        metavar='FILE',
                        help='path to the config file',
                        default='configs/vae.yaml')

    args = parser.parse_args()
    with open(args.filename, 'r') as file:
        try:
            configs = AttrDict(yaml.safe_load(file))
        except yaml.YAMLError as error:
            print(error)

    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda tensor: min_max_normalization(tensor, 0, 1)),
        transforms.Lambda(lambda tensor: tensor_round(tensor))
    ])

    data_dir, model_dir, checkpoint_dir, log_dir = get_platform_path()

    dataset = MNIST(data_dir, transform=img_transform, download=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = configs.model_params.name
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)