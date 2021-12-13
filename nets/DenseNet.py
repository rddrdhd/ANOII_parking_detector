import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models


def conv_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels):
        super(DenseBlock, self).__init__()
        layer = []
        for i in range(num_convs):
            layer.append(
                conv_block(num_channels * i + input_channels, num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for blk in self.net:
            Y = blk(X)
            X = torch.cat((X, Y), dim=1)
        return X


def transition_block(input_channels, num_channels):
    return nn.Sequential(
        nn.BatchNorm2d(input_channels), nn.ReLU(),
        nn.Conv2d(input_channels, num_channels, kernel_size=1),
        nn.AvgPool2d(kernel_size=2, stride=2)
    )


def get_my_densenet(dimensions=3):
    b1 = nn.Sequential(nn.Conv2d(in_channels=dimensions, out_channels=64, kernel_size=7, stride=2, padding=3),
                       nn.BatchNorm2d(num_features=64),
                       nn.ReLU(),
                       nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    num_channels, growth_rate = 64, 32
    num_convs_in_dense_blocks = [4, 4, 4, 4]
    blks = []

    for i, num_convs in enumerate(num_convs_in_dense_blocks):
        blks.append(DenseBlock(num_convs, num_channels, growth_rate))
        # This is the number of output channels in the previous dense block
        num_channels += num_convs * growth_rate
        # A transition layer that halves the number of channels is added between
        # the dense blocks
        if i != len(num_convs_in_dense_blocks) - 1:
            blks.append(transition_block(num_channels, num_channels // 2))
            num_channels = num_channels // 2

    net = nn.Sequential(b1,
                        *blks,
                        nn.BatchNorm2d(num_channels),
                        nn.ReLU(),
                        nn.AdaptiveMaxPool2d((1, 1)),
                        nn.Flatten(),
                        nn.LazyLinear(num_channels, 2)
                        )
    return net


class DenseNet:
    def __init__(self, dimensions):
        self.net = get_my_densenet(dimensions)
