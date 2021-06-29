import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, x):
        x = self.filter(x)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, x):
        x = self.conv_a(x)
        x = self.conv(x)
        return x

class resblock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(resblock, self).__init__()
        self.conv1 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = mfm(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + res
        return out

class network_29layers_v2(nn.Module):
    def __init__(self, block, layers, num_classes=8369):
        super(network_29layers_v2, self).__init__()
        self.conv1    = mfm(3, 48, 5, 1, 2)
        self.block1   = self._make_layer(block, layers[0], 48, 48)
        self.group1   = group(48, 96, 3, 1, 1)
        self.block2   = self._make_layer(block, layers[1], 96, 96)
        self.group2   = group(96, 192, 3, 1, 1)
        self.block3   = self._make_layer(block, layers[2], 192, 192)
        self.group3   = group(192, 128, 3, 1, 1)
        self.block4   = self._make_layer(block, layers[3], 128, 128)
        self.group4   = group(128, 128, 3, 1, 1)
        self.fc       = nn.Linear(4*4*128, 256)
        self.fc2 = nn.Linear(256, num_classes, bias=False)

        in_ch = 256 + 96
        out_ch = 96
        ks = 3
        self.conv_fusion = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=ks, padding=(ks//2))
            
    def _make_layer(self, block, num_blocks, in_channels, out_channels):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x, selected_x = x # 3 x 64 x 64 and 256 x 16 x 16
        x = self.conv1(x) # 3 x 64 x 64 -> 48 x 64 x 64
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2) # 48 x 32 x 32

        x = self.block1(x) # 48 x 32 x 32
        x = self.group1(x) # 96 x 32 x 32
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2) # 96 x 16 x 16

        x = torch.cat([x, selected_x], dim=1) # (96 + 256) x 16 x 16
        x = self.conv_fusion(x) # 96 x 16 x 16

        x = self.block2(x) # 96 x 16 x 16
        x = self.group2(x) # 192 x 16 x 16
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2) # 192 x 8 x 8

        x = self.block3(x) # 192 x 8 x 8
        x = self.group3(x) # 128 x 8 x 8
        x = self.block4(x) # 128 x 8 x 8
        x = self.group4(x) # 128 x 8 x 8
        x = F.max_pool2d(x, 2) + F.avg_pool2d(x, 2) # 128 x 4 x 4

        x = x.view(x.size(0), -1) # 2048
        fc = self.fc(x) # 256
        x = F.dropout(fc, training=self.training)
        out = self.fc2(x)
        return out, fc


def LightCNN_29Layers_v2(**kwargs):
    model = network_29layers_v2(resblock, [1, 2, 3, 4], **kwargs)
    return model

