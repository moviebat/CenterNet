# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# Modified by Xingyi Zhou
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

BN_MOMENTUM = 0.1

model_urls = {
    'mobilenetV2': '/home/zealens/.torch/models/mobilenetv2_1.0-0c6065bc.pth',
}
def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)

class PoseResNet(nn.Module):

    def __init__(self, heads, head_conv, **kwargs):
        self.inplanes = 64
        self.deconv_with_bias = False
        self.heads = heads

        super(PoseResNet, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
        ]

        # building first layer
        width_mult = 1.0
        input_channel = _make_divisible(32 * width_mult, 8)
        layers = [conv_3x3_bn(3, input_channel, 2)]
        # building inverted residual blocks
        block = InvertedResidual
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            layers.append(block(input_channel, output_channel, s, t))
            input_channel = output_channel
            for i in range(1, n):
                layers.append(block(input_channel, output_channel, 1, t))
                input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        #output_channel = _make_divisible(1280 * width_mult, 8) if width_mult > 1.0 else 1280
        #self.conv = conv_1x1_bn(input_channel, output_channel)
        #self.avgpool = nn.AvgPool2d(input_size // 32, stride=1)
        #self.classifier = nn.Linear(output_channel, num_classes)

        #self._initialize_weights()
        # used for deconv layers
        self.inplanes = input_channel
        #self.deconv_layers = self._make_deconv_layer(
        #    1,
        #    [32],
        #    [4],
        #)
        # self.final_layer = []

        for head in sorted(self.heads):
          num_output = self.heads[head]
          if head_conv > 0:
            fc = nn.Sequential(
                nn.Conv2d(24, head_conv,
                  kernel_size=3, padding=1, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_conv, num_output, 
                  kernel_size=1, stride=1, padding=0))
          else:
            fc = nn.Conv2d(
              in_channels=24,
              out_channels=num_output,
              kernel_size=1,
              stride=1,
              padding=0
          )
          self.__setattr__(head, fc)

        # self.final_layer = nn.ModuleList(self.final_layer)

    def forward(self, x):
        x = self.features(x)
        #x = self.deconv_layers(x)

        outs = []

        ret = {}
        for head in self.heads:
            ret[head] = self.__getattr__(head)(x)
        #return [ret]
        outs.append(ret)
        return outs
        #return ret["hm"], ret["wh"], ret["reg"]

    def init_weights(self, pretrained=True):
        if pretrained:
            # print('=> init resnet deconv weights from normal distribution')
            # print('=> init final conv weights from normal distribution')
            for head in self.heads:
              final_layer = self.__getattr__(head)
              for i, m in enumerate(final_layer.modules()):
                  if isinstance(m, nn.Conv2d):
                      # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                      # print('=> init {}.weight as normal(0, 0.001)'.format(name))
                      # print('=> init {}.bias as 0'.format(name))
                      if m.weight.shape[0] == self.heads[head]:
                          if 'hm' in head:
                              nn.init.constant_(m.bias, -2.19)
                          else:
                              nn.init.normal_(m.weight, std=0.001)
                              nn.init.constant_(m.bias, 0)
            #pretrained_state_dict = torch.load(pretrained)
            url = model_urls['mobilenetV2']
            pretrained_state_dict = model_zoo.load_url(url)
            print('=> loading pretrained model {}'.format(url))
            self.load_state_dict(pretrained_state_dict, strict=False)
        else:
            print('=> imagenet pretrained model dose not exist')
            print('=> please download it first')
            raise ValueError('imagenet pretrained model does not exist')
def get_pose_net(heads, head_conv):
  model = PoseResNet(heads, head_conv=head_conv)
  #model.init_weights(pretrained=True)
  return model
