
import torch
import torch.nn as nn
import torch.nn.functional as F


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _  in range(num_convs):
        layers.append(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


def vgg(conv_arch, in_channels):
    # nn.BatchNorm3d(64)
    conv_blks = []
    # The convolutional part
    for (num_convs, out_channels) in conv_arch:
        conv_blks.append(vgg_block(num_convs, in_channels, out_channels))
        in_channels = out_channels

    return nn.Sequential(
        *conv_blks, nn.Flatten(),
        # The fully-connected part
        nn.LazyLinear( 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.LazyLinear( 4096), nn.ReLU(), nn.Dropout(0.5),
        nn.LazyLinear( 10))

def gimme_vggnet(dimensions):
    conv_arch = ((dimensions, 64), (dimensions, 128), (2, 256), (2, 512), (2, 512))
    net = vgg(conv_arch, dimensions)

    return net


class VGGNet:

    def __init__(self, dimensions):
        self.net = gimme_vggnet(dimensions)
