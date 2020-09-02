# @Anshul Paigwar
# Forked from https://github.com/meetshah1995/pytorch-semseg

import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

class conv2DBatchNormRelu(nn.Module):
    def __init__(
        self,
        in_channels,
        n_filters,
        k_size,
        stride,
        padding,
        bias=True,
        dilation=1,
        is_batchnorm=True,
    ):
        super(conv2DBatchNormRelu, self).__init__()

        conv_mod = nn.Conv2d(
            int(in_channels),
            int(n_filters),
            kernel_size=k_size,
            padding=padding,
            stride=stride,
            bias=bias,
            dilation=dilation,
        )

        if is_batchnorm:
            self.cbr_unit = nn.Sequential(
                conv_mod, nn.BatchNorm2d(int(n_filters)), nn.ReLU(inplace=True)
            )
        else:
            self.cbr_unit = nn.Sequential(conv_mod, nn.ReLU(inplace=True))

    def forward(self, inputs):
        outputs = self.cbr_unit(inputs)
        return outputs




class segnetDown2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetDown2, self).__init__()
        self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
        self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        outputs = self.conv2(outputs)
        unpooled_shape = outputs.size()
        outputs, indices = self.maxpool_with_argmax(outputs)
        return outputs, indices, unpooled_shape


# class segnetDown3(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(segnetDown3, self).__init__()
#         self.conv1 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)
#         self.conv2 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
#         self.conv3 = conv2DBatchNormRelu(out_size, out_size, 3, 1, 1)
#         self.maxpool_with_argmax = nn.MaxPool2d(2, 2, return_indices=True)

#     def forward(self, inputs):
#         outputs = self.conv1(inputs)
#         outputs = self.conv2(outputs)
#         outputs = self.conv3(outputs)
#         unpooled_shape = outputs.size()
#         outputs, indices = self.maxpool_with_argmax(outputs)
#         return outputs, indices, unpooled_shape


class segnetUp2(nn.Module):
    def __init__(self, in_size, out_size):
        super(segnetUp2, self).__init__()
        self.unpool = nn.MaxUnpool2d(2, 2)
        self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
        self.conv2 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

    def forward(self, inputs, indices, output_shape):
        outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
        outputs = self.conv1(outputs)
        outputs = self.conv2(outputs)
        return outputs


# class segnetUp3(nn.Module):
#     def __init__(self, in_size, out_size):
#         super(segnetUp3, self).__init__()
#         self.unpool = nn.MaxUnpool2d(2, 2)
#         self.conv1 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
#         self.conv2 = conv2DBatchNormRelu(in_size, in_size, 3, 1, 1)
#         self.conv3 = conv2DBatchNormRelu(in_size, out_size, 3, 1, 1)

#     def forward(self, inputs, indices, output_shape):
#         outputs = self.unpool(input=inputs, indices=indices, output_size=output_shape)
#         outputs = self.conv1(outputs)
#         outputs = self.conv2(outputs)
#         outputs = self.conv3(outputs)
#         return outputs









class segnetGndEst(nn.Module):
    def __init__(self, in_channels=64, is_unpooling=True):
        super(segnetGndEst, self).__init__()

        self.in_channels = in_channels
        self.is_unpooling = is_unpooling

        self.down1 = segnetDown2(self.in_channels, 128)
        self.down2 = segnetDown2(128, 256)

        self.up2 = segnetUp2(256, 128)
        self.up1 = segnetUp2(128, 64)

        self.regressor = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, inputs):

        down1, indices_1, unpool_shape1 = self.down1(inputs)
        down2, indices_2, unpool_shape2 = self.down2(down1)

        up2 = self.up2(down2, indices_2, unpool_shape2)
        up1 = self.up1(up2, indices_1, unpool_shape1)
        gnd_pred = self.regressor(up1)

        return gnd_pred


# class segnet(nn.Module):
#     def __init__(self, n_classes=21, in_channels=3, is_unpooling=True):
#         super(segnet, self).__init__()

#         self.in_channels = in_channels
#         self.is_unpooling = is_unpooling

#         self.down1 = segnetDown2(self.in_channels, 64)
#         self.down2 = segnetDown2(64, 128)
#         self.down3 = segnetDown3(128, 256)
#         self.down4 = segnetDown3(256, 512)
#         self.down5 = segnetDown3(512, 512)

#         self.up5 = segnetUp3(512, 512)
#         self.up4 = segnetUp3(512, 256)
#         self.up3 = segnetUp3(256, 128)
#         self.up2 = segnetUp2(128, 64)
#         self.up1 = segnetUp2(64, n_classes)

#     def forward(self, inputs):

#         down1, indices_1, unpool_shape1 = self.down1(inputs)
#         down2, indices_2, unpool_shape2 = self.down2(down1)
#         down3, indices_3, unpool_shape3 = self.down3(down2)
#         down4, indices_4, unpool_shape4 = self.down4(down3)
#         down5, indices_5, unpool_shape5 = self.down5(down4)

#         up5 = self.up5(down5, indices_5, unpool_shape5)
#         up4 = self.up4(up5, indices_4, unpool_shape4)
#         up3 = self.up3(up4, indices_3, unpool_shape3)
#         up2 = self.up2(up3, indices_2, unpool_shape2)
#         up1 = self.up1(up2, indices_1, unpool_shape1)

#         return up1
