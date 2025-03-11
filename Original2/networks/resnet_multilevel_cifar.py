
import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import relu




def normalize(x:torch.Tensor) -> torch.Tensor:
    x_norm = torch.norm(x, p=2, dim=1).unsqueeze(1).expand_as(x)
    x_normalized = x.div(x_norm + 0.00001)
    return x_normalized


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1) -> F.conv2d:
    """
    Instantiates a 3x3 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :param groups: number of groups for group convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, groups=groups, bias=False)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> F.conv2d:
    """
    Instantiates a 1x1 convolutional layer with no bias.
    :param in_planes: number of input channels
    :param out_planes: number of output channels
    :param stride: stride of the convolution
    :return: convolutional layer
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, 
                     padding=0, bias=False)



class BasicBlock(nn.Module):
    """
    The basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        """
        Instantiates the basic block of the network.
        :param in_planes: the number of input channels
        :param planes: the number of channels (to be possibly expanded)
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute a forward pass.
        :param x: input tensor (batch_size, input_size)
        :return: output tensor (10)
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class DownConv(nn.Module):
    """
    Convolutional downsampling block
    """
    def __init__(self, channel_in, channel_out):
        super(DownConv, self).__init__()
        self.op = nn.Sequential(
            conv3x3(channel_in, channel_in, stride=2, groups=channel_in),
            conv1x1(channel_in, channel_in, stride=1),
            nn.BatchNorm2d(channel_in),
            nn.ReLU(),
            conv3x3(channel_in, channel_in, stride=1, groups=channel_in),
            conv1x1(channel_in, channel_out, stride=1),
            nn.BatchNorm2d(channel_out),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.op(x)


class ResNetSD(nn.Module):

    def __init__(self, block: BasicBlock, num_blocks: List[int],
                 num_classes: int, nf: int) -> None:
        """
        Instantiates the layers of the network.
        :param block: the basic ResNet block
        :param num_blocks: the number of blocks per layer
        :param num_classes: the number of output classes
        :param nf: the number of filters
        """
        super(ResNetSD, self).__init__()

        # 蒸留損失における生徒出力の番号
        self.expert_number = 3

        # resnet18の基本的な構造
        self.in_planes = nf
        self.block = block
        self.num_classes = num_classes
        self.nf = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)

        # 各層の出力からクラス分類を行うための線形層（次元数変化の例：512 -> 100）
        self.linear = nn.ModuleList([
            nn.Linear(nf * 8 * block.expansion, num_classes),
            nn.Linear(nf * 8 * block.expansion, num_classes),
            nn.Linear(nf * 8 * block.expansion, num_classes),
            nn.Linear(nf * 8 * block.expansion, num_classes)
        ])

        # 各層の出力から対照損失を計算するための線形層（次元数変化の例：512 -> 128）
        self.simclr = nn.ModuleList([
            nn.Linear(nf * 8 * block.expansion, 128),
            nn.Linear(nf * 8 * block.expansion, 128),
            nn.Linear(nf * 8 * block.expansion, 128),
            nn.Linear(nf * 8 * block.expansion, 128)
        ])

        # 各層の出力を同じ特徴次元数にするための畳み込み層
        self.feature_alignment_layer = nn.ModuleList([
            self._make_feature_layer(nf * 1, nf * 8 * block.expansion),
            self._make_feature_layer(nf * 2, nf * 8 * block.expansion),
            self._make_feature_layer(nf * 4, nf * 8 * block.expansion),
            self._make_feature_layer(nf * 8, nf * 8 * block.expansion)
        ])

        # 重要な位置の特定？？？（よくわからない）
        self.attention_layers = nn.ModuleList([
            self._make_attention_layer(nf * 1 * block.expansion),
            self._make_attention_layer(nf * 2 * block.expansion),
            self._make_attention_layer(nf * 4 * block.expansion),
        ])

        # 
        self.final_addaption_layer = nn.Linear(nf * 8 * block.expansion, nf * 8 * block.expansion)

        # 
        self.moe_layer = nn.Linear(4 * nf * 8 * block.expansion, num_classes)


        # 初期化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    

    def _make_feature_layer(self, in_planes, out_planes):
        feature_layer = []
        num_down = out_planes // in_planes
        num_down = int(math.log(num_down, 2))
        for i in range(num_down):
            feature_layer.append(DownConv(in_planes, in_planes*2))
            in_planes = in_planes * 2
        assert in_planes == out_planes
        feature_layer.append(nn.AdaptiveAvgPool2d((1, 1)))
        return nn.Sequential(*feature_layer)
    
    def _make_attention_layer(self, planes):
        layers = [
            DownConv(planes, planes),
            nn.BatchNorm2d(planes),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Sigmoid()
        ]
        return nn.Sequential(*layers)

    def _make_layer(self, block: BasicBlock, planes: int,
                    num_blocks: int, stride: int) -> nn.Module:
        """
        Instantiates a ResNet layer.
        :param block: ResNet basic block
        :param planes: channels across the network
        :param num_blocks: number of blocks
        :param stride: stride
        :return: ResNet layer
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    

    def features(self, x: torch.Tensor) -> List[torch.Tensor]:

        out = relu(self.bn1(self.conv1(x)))

        out = relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        fea1 = self.attention_layers[0](out)
        fea1 = fea1 * out
        fea1 = self.feature_alignment_layer[0](fea1).view(out.size(0), -1)

        out = self.layer2(out)
        fea2 = self.attention_layers[1](out)
        fea2 = fea2 * out
        fea2 = self.feature_alignment_layer[1](fea2).view(out.size(0), -1)

        out = self.layer3(out)
        fea3 = self.attention_layers[2](out)
        fea3 = fea3 * out
        fea3 = self.feature_alignment_layer[2](fea3).view(out.size(0), -1)

        out = self.layer4(out)
        fea4 = self.feature_alignment_layer[3](out).view(out.size(0), -1)

        return [fea1, fea2, fea3, fea4]

    def head(self, x: List[torch.Tensor], use_proj=False):

        if use_proj:
            out_list = [self.simclr[i](x[i]) for i in range(len(x))]
        else:
            out_list = [self.linear[i](x[i]) for i in range(len(x))]
        return out_list



    def forward(self, x: torch.tensor, use_proj=False):

        assert False




def resnet18_sd(nclasses: int, nf: int = 64) -> ResNetSD:
    """
    Instantiates a ResNet18 network.
    :param nclasses: number of output classes
    :param nf: number of filters
    :return: ResNet network
    """
    return ResNetSD(BasicBlock, [2, 2, 2, 2], nclasses, nf)