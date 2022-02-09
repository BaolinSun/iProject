import torch
import torch.nn as nn
import cv2

from model.heads.ins_his_head import INS_HIS_HEAD
from model.heads.mask_feat_head import MaskFeatHead
from model.backbones.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from model.necks.fpn import FPN


class INS_HIS(nn.Module):

    def __init__(self, cfg=None, pretrained=None, mode='trian'):
        super(INS_HIS, self).__init__()

        # @backbone
        self.backbone = resnet18(pretrained=True,
                                 loadpath='checkpoints/resnet18-5c106cde.pth')

        # @neck
        self.fpn = FPN(in_channels=[64, 128, 256, 512],
                       out_channels=256,
                       start_level=0,
                       num_outs=5,
                       upsample_cfg=dict(mode='nearest'))

        self.mask_feat_head = MaskFeatHead(in_channels=256,
                                        out_channels=128,
                                        start_level=0,
                                        end_level=3,
                                        num_classes=128)
        self.bbox_head = INS_HIS_HEAD(num_classes=2,
                                      in_channels=256,
                                      seg_feat_channels=256,
                                      stacked_convs=2,
                                      num_grids=[40, 36, 24, 16, 12],
                                      ins_out_channels=128)

    def extract_feat(self, img):
        x = self.backbone(img)
        x = self.fpn(x)
        return x

    def forward(self, img):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        mask_feat_pred = self.mask_feat_head(x[self.mask_feat_head.start_level:self.mask_feat_head.end_level+1])

        return x


if __name__ == '__main__':
    img = cv2.imread('tmp.jpg')
    img = img.transpose(2, 0, 1)
    img = torch.tensor(img, dtype=torch.float32)
    # img = torch.from_numpy(img)
    # img = img.type(torch.)
    img = img.unsqueeze(0)
    img = img.cuda()

    model = INS_HIS()
    model = model.cuda()

    x = model.forward(img)

    print('success...')