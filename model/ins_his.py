from pip import main
import torch
import torch.nn as nn

from backbones.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from necks.fpn import FPN


class INS_HIS(nn.Module):

    def __init__(self, cfg=None, pretrained=None, mode='trian'):
        super(INS_HIS, self).__init__()

        # @backbone
        self.backbone = resnet18(
            pretrained=True, loadpath='../checkpoints/resnet18-5c106cde.pth')

        # @neck
        self.fpn = FPN(in_channels=[256, 128, 256, 512],
                       out_channels=256,
                       start_level=0,
                       num_outs=5,
                       upsample_cfg=dict(mode='nearest'))


if __name__ == '__main__':
    model = INS_HIS()
    print(model)