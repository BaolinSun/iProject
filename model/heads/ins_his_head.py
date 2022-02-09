'''
Author: sunbaolin
Date: 2022-02-04 18:16:07
LastEditors: sunbaolin
LastEditTime: 2022-02-07 18:42:33
Description: file content
FilePath: /iProject/model/heads/ins_his_head.py
'''

from statistics import mode
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.nninit import normal_init, bias_init_with_prob
from model.misc import multi_apply, points_nms

class INS_HIS_HEAD(nn.Module):

    def __init__(
            self,
            num_classes,  # dataset classes number
            in_channels,  # 256 fpn output
            seg_feat_channels=256,  # seg feature channels
            stacked_convs=4,
            num_grids=None,
            ins_out_channels=64):
        super(INS_HIS_HEAD, self).__init__()
        self.num_classes = num_classes
        self.seg_num_grids = num_grids
        self.cate_out_channels = self.num_classes - 1
        self.ins_out_channels = ins_out_channels
        self.in_channels = in_channels
        self.seg_feat_channels = seg_feat_channels
        self.stacked_convs = stacked_convs
        self.kernel_out_channels = self.ins_out_channels * 1 * 1

        self._init_layers()

    def _init_layers(self):
        norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
        self.cate_convs = nn.ModuleList()
        self.kernel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            # 第0层加上位置信息，x,y两个通道，cat到卷积输出上
            chn = self.in_channels + 2 if i == 0 else self.seg_feat_channels
            self.kernel_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn,
                              self.seg_feat_channels,
                              3,
                              stride=1,
                              padding=1,
                              bias=norm_cfg is None),
                    nn.GroupNorm(num_channels=self.seg_feat_channels,
                                 num_groups=32), nn.ReLU(inplace=True)))

            chn = self.in_channels if i == 0 else self.seg_feat_channels
            self.cate_convs.append(
                nn.Sequential(
                    nn.Conv2d(chn,
                              self.seg_feat_channels,
                              3,
                              stride=1,
                              padding=1,
                              bias=norm_cfg is None),
                    nn.GroupNorm(num_channels=self.seg_feat_channels,
                                 num_groups=32), nn.ReLU(inplace=True)))

            self.solo_cate = nn.Conv2d(self.seg_feat_channels,
                                       self.cate_out_channels,
                                       3,
                                       padding=1)
            self.solo_kernel = nn.Conv2d(self.seg_feat_channels,
                                         self.kernel_out_channels,
                                         3,
                                         padding=1)

    def init_weights(self):
        for m in self.cate_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        for m in self.kernel_convs:
            if isinstance(m, nn.Sequential):
                for con in m:
                    if isinstance(con, nn.Conv2d):
                        normal_init(con, std=0.01)

        bias_cate = bias_init_with_prob(0.01)
        normal_init(self.solo_cate, std=0.01, bias=bias_cate)
        normal_init(self.solo_kernel, std=0.01)

    def forward(self, feats, eval=False):
        new_feats = self.split_feats(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in new_feats]
        upsampled_size = (featmap_sizes[0][0]*2, featmap_sizes[0][1]*2)
        cate_pred, kernel_pred = multi_apply(self.forward_single, new_feats, list(range(len(self.seg_num_grids))), eval=eval, upsampled_size=upsampled_size)

        return cate_pred, kernel_pred

    def forward_single(self, x, idx, eval=False, upsampled_size=None):
        ins_kernel_feat = x

        x_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-1], device=ins_kernel_feat.device)
        y_range = torch.linspace(-1, 1, ins_kernel_feat.shape[-2], device=ins_kernel_feat.device)
        y, x = torch.meshgrid(y_range, x_range)
        y = y.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        x = x.expand([ins_kernel_feat.shape[0], 1, -1, -1])
        coord_feat = torch.cat([x, y], 1)
        ins_kernel_feat = torch.cat([ins_kernel_feat, coord_feat], 1)

        kernel_feat = ins_kernel_feat
        seg_num_grid = self.seg_num_grids[idx]
        kernel_feat = F.interpolate(kernel_feat, size=seg_num_grid, mode='bilinear', align_corners=False)

        cate_feat = kernel_feat[:, :-2, :, :]

        kernel_feat = kernel_feat.contiguous()
        for i, kernel_layer in enumerate(self.kernel_convs):
            kernel_feat = kernel_layer(kernel_feat)
        kernel_pred = self.solo_kernel(kernel_feat)

        cate_feat = cate_feat.contiguous()
        for i, cate_layer in enumerate(self.cate_convs):
            cate_feat = cate_layer(cate_feat)
        cate_pred = self.solo_cate(cate_feat)

        if eval:
            cate_pred = points_nms(cate_pred.sigmoid(), kernel=2).premute(0, 2, 3, 1)

        return cate_pred, kernel_pred

    def split_feats(self, feats):
        return (F.interpolate(feats[0],
                              scale_factor=0.5,
                              mode='bilinear',
                              align_corners=False,
                              recompute_scale_factor=True),
                feats[1],
                feats[2],
                feats[3],
                F.interpolate(feats[4],
                              size=feats[3].shape[-2:],
                              mode='bilinear',
                              align_corners=False))

    def loss(self, cate_preds, kernel_preds, ins_pred, gt_bbox_list, gt_label_list, gt_mask_list):
        
        mask_feat_size = ins_pred.size()[-2:]
