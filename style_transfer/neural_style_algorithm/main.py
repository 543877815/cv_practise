# borrow heavily from https://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# paper: A Neural Algorithm of Artistic Style, https://arxiv.org/abs/1508.06576

from __future__ import print_function
import torch
import torch.nn as nn
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

from style_transfer.utils import StyleLoss, ContentLoss, Normalization, imshow, save
import torchvision.transforms as transforms
import torchvision.models as models

import copy
from utils import get_platform_path
import argparse


def get_style_model_and_losses(cnn, normalization_mean, normalization_std,
                               style_img, content_img, style_layers=None, content_layers=None):
    cnn = copy.deepcopy(cnn)

    # normalization module
    normalization = Normalization(normalization_mean, normalization_std).to(device)

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential(normalization)

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ContentLoss
            # and StyleLoss we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        # 将 ContentLoss 和 StyleLoss 整合到 models 中
        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


def run_style_transfer(model, normalization_mean, normalization_std,
                       content_img, style_img, input_img, style_layers,
                       content_layers, epochs=300,
                       style_weight=1000000, content_weight=10):
    """Run the style transfer."""
    print('Building the style transfer models..')
    model, style_losses, content_losses = get_style_model_and_losses(model, normalization_mean, normalization_std,
                                                                     style_img, content_img, style_layers,
                                                                     content_layers)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print('Optimizing..')
    run = [0]
    while run[0] <= epochs:

        def closure():
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img.data.clamp_(0, 1)

    return input_img


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pytorch super resolution example')
    # cuda configuration
    parser.add_argument('--use_cuda', type=bool, default=True, help='whether to use cuda')

    # data configuration
    parser.add_argument('--width', type=int, default=512, help="desired size of the output image")
    parser.add_argument('--height', type=int, default=512, help="desired size of the output image")
    parser.add_argument('--style_img', type=str, default="../images/style/first.jpg", help="style image file name")
    parser.add_argument('--content_img', type=str, default='../images/content/first.jpg', help="content image file name")
    parser.add_argument('--output_img', type=str, default='../images/output/first.jpg', help="output image file name")

    # setting
    parser.add_argument('--epochs', type=int, default=300, help="desired epochs to run")
    args = parser.parse_args()


    # detect device
    print("CUDA Available: ", torch.cuda.is_available())
    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")

    # data preparing
    data_transform = transforms.Compose([
        transforms.Resize([args.height, args.width]),
        transforms.ToTensor()
    ])


    def image_loader(image_name):
        image = Image.open(image_name)
        # fake batch dimension required to fit network's input dimensions
        image = data_transform(image).unsqueeze(0)
        return image.to(device, torch.float)


    style_img = image_loader(args.style_img)
    content_img = image_loader(args.content_img)

    assert style_img.size() == content_img.size(), "we need to import style and content image of the same size"

    # models
    model = models.vgg19(pretrained=True).features.to(device).eval()

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

    # desired depth layers to compute style/content losses :
    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    # init image, if using noise, the content weight should be 10 or larger, else first is enough
    # input_img = content_img.clone()
    input_img = torch.randn(content_img.data.size(), device=device)

    # run
    output = run_style_transfer(model=model, normalization_mean=cnn_normalization_mean,
                                normalization_std=cnn_normalization_std,
                                content_img=content_img, style_img=style_img, input_img=input_img,
                                content_layers=content_layers_default, style_layers=style_layers_default,
                                epochs=args.epochs)

    plt.figure()
    imshow(output, title='Output Image')

    # sphinx_gallery_thumbnail_number = 4
    plt.ioff()
    plt.show()

    # save result
    save(output, file=args.output_img)
