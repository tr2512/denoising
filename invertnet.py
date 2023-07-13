import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super(ResBlock, self).__init__()
        feature = 64
        self.conv1 = nn.Conv2d(channel_in, feature, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv2d(feature, feature, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d((feature+channel_in), channel_out, kernel_size=3, padding=1)

    def forward(self, x):
        residual = self.relu1(self.conv1(x))
        residual = self.relu1(self.conv2(residual))
        input = torch.cat((x, residual), dim=1)
        out = self.conv3(input)
        return out


class InvBlockExp(nn.Module):
    def __init__(self, channel_num):
        super(InvBlockExp, self).__init__()
        self.split_len1 = 3
        self.split_len2 = channel_num - self.split_len1

        self.F = ResBlock(self.split_len2, self.split_len1)
        self.G = ResBlock(self.split_len1, self.split_len2)
        self.H = ResBlock(self.split_len1, self.split_len2)

    def forward(self, x, reverse=False):
        assert x.shape[1] == self.split_len1 + self.split_len2
        x1, x2 = x[:, :self.split_len1, :, :], x[:, self.split_len1:, :, :]

        if not reverse:
            y1 = x1 + self.F(x2)
            self.s = torch.sigmoid(self.H(y1)) * 2 - 1
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = torch.sigmoid(self.H(x1)) * 2 - 1
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), dim=1)

class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * channel_in, dim=0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, reverse=False):
        if not reverse:
            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            return out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
        else:
            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

class InvNet(nn.Module):
    def __init__(self, num_downscale=2, num_blocks=8):
        super(InvNet, self).__init__()

        operations = []

        current_channel = 3
        for i in range(num_downscale):
            b = HaarDownsampling(current_channel)
            operations.append(b)
            current_channel *= 4
            for j in range(num_blocks):
                b = InvBlockExp(current_channel)
                operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, reverse=False):

        if not reverse:
            for op in self.operations:
                x = op.forward(x, reverse)
        else:
            for op in reversed(self.operations):
                x = op.forward(x, reverse)

        return x
