import sys

from .attention import *
from .densenet import *
from .googlenet import *
from .inceptionv3 import *
from .inceptionv4 import *
from .mobilenet import *
from .mobilenetv2 import *
from .nasnet import *
from .preactresnet import *
from .resnet import *
from .resnext import *
from .rir import *
from .senet import *
from .shufflenet import *
from .shufflenetv2 import *
from .squeezenet import *
from .vgg import *
from .wideresidual import *
from .xception import *


def get_network(args):
    if args.net == 'vgg16':
        net = vgg16_bn()
    elif args.net == 'vgg13':
        net = vgg13_bn()
    elif args.net == 'vgg11':
        net = vgg11_bn()
    elif args.net == 'vgg19':
        net = vgg19_bn()
    elif args.net == 'densenet121':
        net = densenet121()
    elif args.net == 'densenet161':
        net = densenet161()
    elif args.net == 'densenet169':
        net = densenet169()
    elif args.net == 'densenet201':
        net = densenet201()
    elif args.net == 'googlenet':
        net = googlenet()
    elif args.net == 'inceptionv3':
        net = inceptionv3()
    elif args.net == 'inceptionv4':
        net = inceptionv4()
    elif args.net == 'inceptionresnetv2':
        net = inception_resnet_v2()
    elif args.net == 'xception':
        net = xception()
    elif args.net == 'resnet18':
        net = resnet18()
    elif args.net == 'resnet34':
        net = resnet34()
    elif args.net == 'resnet50':
        net = resnet50()
    elif args.net == 'resnet101':
        net = resnet101()
    elif args.net == 'resnet152':
        net = resnet152()
    elif args.net == 'preactresnet18':
        net = preactresnet18()
    elif args.net == 'preactresnet34':
        net = preactresnet34()
    elif args.net == 'preactresnet50':
        net = preactresnet50()
    elif args.net == 'preactresnet101':
        net = preactresnet101()
    elif args.net == 'preactresnet152':
        net = preactresnet152()
    elif args.net == 'resnext50':
        net = resnext50()
    elif args.net == 'resnext101':
        net = resnext101()
    elif args.net == 'resnext152':
        net = resnext152()
    elif args.net == 'shufflenet':
        net = shufflenet()
    elif args.net == 'shufflenetv2':
        net = shufflenetv2()
    elif args.net == 'squeezenet':
        net = squeezenet()
    elif args.net == 'mobilenet':
        net = mobilenet()
    elif args.net == 'mobilenetv2':
        net = mobilenetv2()
    elif args.net == 'nasnet':
        net = nasnet()
    elif args.net == 'attention56':
        net = attention56()
    elif args.net == 'attention92':
        net = attention92()
    elif args.net == 'seresnet18':
        net = seresnet18()
    elif args.net == 'seresnet34':
        net = seresnet34()
    elif args.net == 'seresnet50':
        net = seresnet50()
    elif args.net == 'seresnet101':
        net = seresnet101()
    elif args.net == 'seresnet152':
        net = seresnet152()
    elif args.net == 'wideresnet':
        net = wideresnet()
    else:
        print('the network name you have entered is not supported yet')
        sys.exit()
    return net
