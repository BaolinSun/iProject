'''
    @author: sunbaolin
    @contact: baolin.sun@mail.sdu.edu.cn
    @time: 2022.01.21
    @file: resnet.py
    @desc: resnet backbone
'''

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

# Resnet设计重要原则：当feature map大小降低一半时，feature map的数量增加一倍，这保持了网络层的复杂度
__all__ = [
    'Resnet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152'
]


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    '''
    @ groups 分组卷积参数
    @ dilation 空洞卷积
    '''
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)


# Note: bias设置为False，原因是使用了Batch Normalization，而其对隐藏层有去均值的操作，所以这里常数项可以消去
# BasicBlocks是为resnet18及resnet34设计的，由于较浅的结构可以不使用Bottleneck
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super().__init__()

        # BatchNorm2d最常用于卷积网络中，防止梯度爆炸或消失，设置的参数就是卷积的输出通道数
        # 计算各个维度的标准和方差，进行归一化操作
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 note supported in BasicBlock')

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)  # 卷积操作， 输入通道，输出通道，步长
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)  # conv层通道数都是64的倍数
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    # Note: 两个3*3结构为主加上bn层和一次relu激活
    # downsample是由于x+out的操作，需要对原始输入的x进行downsample
    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 当连接的维度不同时，使用1x1卷积核讲低维转成高维，才能进行相加
        if self.downsample is not None:
            identity = self.downsample(x)  # 下采样

        out += identity  # 残差网络
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4  # expansion 是对输出通道数的倍乘，一层里面最终输出是四倍膨胀

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 downsample=None,
                 groups=1,
                 base_width=64,
                 dilation=1,
                 norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        width = int(planes * (base_width / 64.)) * groups

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # 不管是BasicBlock还是BottleNeck，最后都会做一个判断是否需要给x做downsample，因为必须要把x的通道数变成于主枝的输出通道一致
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_classes=1000,
                 zero_init_residual=False,
                 groups=1,
                 width_per_group=64,
                 replace_stride_with_dilation=None,
                 norm_layer=None):
        super().__init__()

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.inplanes = 64  # 设置默认输入通道
        self.dilation = 1

        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(
                                 replace_stride_with_dilation))

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3,
            bias=False)  # 输入3  输出inplanes  步长为2  填充为3   偏移量为false
        self.bn1 = norm_layer(self.inplanes)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])

        # 对卷积层和BN层初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self._freeze_stages()
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _freeze_stages(self):
        self.bn1.eval()
        for m in [self.conv1, self.bn1]:
            for param in m.parameters():
                param.requires_grad = False
        for i in range(1, 1 + 1):
            m = getattr(self, 'layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        """
        Args:
            @block: 选择使用的residual结构是BasicBlock还是BottleNeck
            @planes: 基准通道数
            @blocks: 每个blocks包含多少个residual子结构
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dialtion = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1

        # 如果stride不等于1或者维度不匹配的时候的downsample，用1*1卷积的操作来进行升维，然后对其进行一次BN操作
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion))

        layers = [
        ]  # [3，4，6，3]表示按次序生成3个Bottleneck，4个Bottleneck，6个Bottleneck，3个Bottleneck
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dialtion, norm_layer))
        # 这里分两个block是因为要将一整个Lyaer进行output size那里，维度是依次下降两倍的，第一个是设置了stride=2所以维度下降一半，剩下的不需要进行维度下降，都是一样的维度
        self.inplanes = planes * block.expansion
        for _ in range(
                1, blocks
        ):  # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造
            layers.append(
                block(self.inplanes,
                      planes,
                      groups=self.groups,
                      base_width=self.base_width,
                      dilation=self.dilation,
                      norm_layer=norm_layer))

        return nn.Sequential(*layers)

    # ResNet 共有五个阶段，其中第一阶段为一个 7*7 的卷积，stride = 2，padding = 3，然后经过 BN、ReLU 和 maxpooling，此时特征图的尺寸已成为输入的 1/4
    # 接下来是四个阶段，也就是代码中 layer1，layer2，layer3，layer4。这里用 _make_layer 函数产生四个 Layer，需要用户输入每个 layer 的 block 数目（ 即layers列表 )以及采用的 block 类型（基础版 BasicBlock 还是 Bottleneck 版）
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 第一阶段进行普通卷积 变成原来1/4

        # 其实所谓的layer1，2，3，4都是由不同参数的_make_layer()方法得到的。看_make_layer()的参数，发现了layers[0~3]就是上面输入的[3，4，6，3]，即layers[0]是3，layers[1]是4，layers[2]是6，layers[3]是3。
        outs = []
        x = self.layer1(x)
        outs.append(x)
        x = self.layer2(x)
        outs.append(x)
        x = self.layer3(x)
        outs.append(x)
        x = self.layer4(x)
        outs.append(x)

        return tuple(outs)

    def train(self, mode=True):
        self._freeze_stages()
        if mode:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


def _resnet(arch, block, layers, pretrained, loadpath, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)

    if pretrained and loadpath is not None:
        state_dict = torch.load(loadpath)
        model.load_state_dict(state_dict, strict=False)

    return model


def resnet18(pretrained=False, loadpath=None, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, loadpath,
                   progress, **kwargs)


def resnet34(pretrained=False, loadpath=None, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, loadpath,
                   progress, **kwargs)


def resnet50(pretrained=False, loadpath=None, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, loadpath,
                   progress, **kwargs)


def resnet101(pretrained=False, loadpath=None, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained,
                   loadpath, progress, **kwargs)


def resnet152(pretrained=False, loadpath=None, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained,
                   loadpath, progress, **kwargs)


if __name__ == '__main__':
    backbone = resnet50(pretrained=True,
                        loadpath='../checkpoints/resnet50-19c8e357.pth')
    print(backbone)
