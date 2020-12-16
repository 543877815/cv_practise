import sys

from .vgg import *
from .dpn import *
from .lenet import *
from .senet import *
from .pnasnet import *
from .densenet import *
from .googlenet import *
from .shufflenet import *
from .shufflenetv2 import *
from .resnet import *
from .resnext import *
from .preact_resnet import *
from .mobilenet import *
from .mobilenetv2 import *
from .efficientnet import *


def get_network(args):
    if args.net == 'vgg11':
        net = VGG('VGG11')
    elif args.net == 'vgg13':
        net = VGG('VGG13')
    elif args.net == 'vgg16':
        net = VGG('VGG16')
    elif args.net == 'vgg19':
        net = VGG('VGG19')
    elif args.net == 'DenseNet121':
        net = DenseNet121()
    elif args.net == 'DenseNet161':
        net = DenseNet161()
    elif args.net == 'DenseNet169':
        net = DenseNet169()
    elif args.net == 'DenseNet201':
        net = DenseNet201()
    elif args.net == 'GoogLeNet':
        net = GoogLeNet()
    elif args.net == 'DPN26':
        net = DPN26()
    elif args.net == 'DPN92':
        net = DPN92()
    elif args.net == 'LeNet':
        net = LeNet()
    elif args.net == 'SENet18':
        net = SENet18()
    elif args.net == 'PNASNetA':
        net = PNASNetA()
    elif args.net == 'PNASNetB':
        net = PNASNetB()
    elif args.net == 'ShuffleNetG2':
        net = ShuffleNetG2()
    elif args.net == 'ShuffleNetG3':
        net = ShuffleNetG3()
    elif args.net == 'ShuffleNetV2':
        net = ShuffleNetV2(net_size=0.5)
    elif args.net == 'resnet18':
        net = ResNet18()
    elif args.net == 'resnet34':
        net = ResNet34()
    elif args.net == 'resnet50':
        net = ResNet50()
    elif args.net == 'resnet101':
        net = ResNet101()
    elif args.net == 'resnet152':
        net = ResNet152()
    elif args.net == 'ResNeXt29_2x64d':
        net = ResNeXt29_2x64d()
    elif args.net == 'ResNeXt29_4x64d':
        net = ResNeXt29_4x64d()
    elif args.net == 'ResNeXt29_8x64d':
        net = ResNeXt29_8x64d()
    elif args.net == 'ResNeXt29_32x4d':
        net = ResNeXt29_32x4d()
    elif args.net == 'PreActResNet18':
        net = PreActResNet18()
    elif args.net == 'PreActResNet34':
        net = PreActResNet34()
    elif args.net == 'PreActResNet50':
        net = PreActResNet50()
    elif args.net == 'PreActResNet101':
        net = PreActResNet101()
    elif args.net == 'PreActResNet152':
        net = PreActResNet152()
    elif args.net == 'MobileNet':
        net = MobileNet()
    elif args.net == 'MobileNetV2':
        net = MobileNetV2()
    elif args.net == 'EfficientNetB0':
        net = EfficientNetB0()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net
