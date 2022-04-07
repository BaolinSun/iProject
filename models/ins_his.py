'''
Author: sunbaolin
Date: 2022-02-11 11:19:24
LastEditors: sunbaolin
LastEditTime: 2022-02-23 16:34:06
FilePath: /iProject/models/ins_his.py
Description: 

'''
import torch
import torch.nn as nn
import cv2

from models.heads.mask_kernel_head import MaskKernelHead
from models.heads.mask_feat_head import MaskFeatHead
from models.backbones.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from models.necks.fpn import FPN


class INS_HIS(nn.Module):
    
    def __init__(self,
                 cfg=None,
                 pretrained=None,
                 mode='train'):
        super(INS_HIS, self).__init__()
        # @backbone
        if cfg.backbone.name == 'resnet18':
            self.backbone = resnet18(pretrained=True, loadpath = cfg.backbone.path)
        elif cfg.backbone.name == 'resnet34':
            self.backbone = resnet34(pretrained=True, loadpath = cfg.backbone.path)
        elif cfg.backbone.name == 'resnet50':
            self.backbone = resnet50(pretrained=True, loadpath = cfg.backbone.path)
        elif cfg.backbone.name == 'resnet101':
            self.backbone = resnet101(pretrained=True, loadpath = cfg.backbone.path)
        elif cfg.backbone.name == 'resnet152':
            self.backbone = resnet152(pretrained=True, loadpath = cfg.backbone.path)
        else:
            raise NotImplementedError
        
        # @neck
        self.fpn = FPN(in_channels=cfg.fpn.in_channels,
                       out_channels=cfg.fpn.out_channels,
                       start_level=cfg.fpn.start_level,
                       num_outs=cfg.fpn.num_outs,
                       upsample_cfg=dict(mode='nearest'))

        #this set only support resnet18 and resnet34 backbone
        self.mask_feat_head = MaskFeatHead(in_channels=256,
                            out_channels=128,
                            start_level=0,
                            end_level=3,

                            num_classes=cfg.kernel_head.num_classes)
        #this set only support resnet18 and resnet34 backbone
        self.bbox_head = MaskKernelHead(num_classes=2,
                            in_channels=256,
                            seg_feat_channels=cfg.kernel_head.seg_feat_channels,
                            stacked_convs=cfg.kernel_head.stacked_convs,
                            strides=[8, 8, 16, 32, 32],
                            scale_ranges=cfg.kernel_head.scale_ranges,
                            num_grids=[40, 36, 24, 16, 12],
                            ins_out_channels=cfg.kernel_head.ins_out_channels)
        
        self.mode = mode

        self.test_cfg = cfg.test_cfg

        if self.mode == 'train':
            self.backbone.train(mode=True)
        else:
            self.backbone.train(mode=True)
        
        if pretrained is None:
            self.init_weights() #if first train, use this initweight
        else:
            self.load_weights(pretrained)             #load weight from file

        print('#@backbone:', cfg.backbone.name, '#@neck:', cfg.fpn.name, '#@head:', cfg.name)
    
    def init_weights(self):
        #fpn
        if isinstance(self.fpn, nn.Sequential):
            for m in self.fpn:
                m.init_weights()
        else:
            self.fpn.init_weights()
        
        #mask feature mask
        if isinstance(self.mask_feat_head, nn.Sequential):
            for m in self.mask_feat_head:
                m.init_weights()
        else:
            self.mask_feat_head.init_weights()

        self.bbox_head.init_weights()
    
    def save_weights(self, path):
        """ Saves the model's weights using compression because the file sizes were getting too big. """
        torch.save(self.state_dict(), path)

    def load_weights(self, path):
        state_dict = torch.load(path)
        self.load_state_dict(state_dict)
 
    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        x = self.fpn(x)
        return x
    
    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs


    def forward(self, img, img_meta, return_loss=True, **kwargs):
        """
        Calls either forward_train or forward_test depending on whether
        return_loss=True. Note this setting will change the expected inputs.
        When `return_loss=True`, img and img_meta are single-nested (i.e.
        Tensor and List[dict]), and when `resturn_loss=False`, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_meta, **kwargs)
        else:
            return self.forward_test(img, img_meta, **kwargs)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None):


        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        mask_feat_pred = self.mask_feat_head(x[self.mask_feat_head.start_level:self.mask_feat_head.end_level + 1])
        loss_inputs = outs + (mask_feat_pred, gt_bboxes, gt_labels, gt_masks, img_metas)

        losses = self.bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

  
    # 短边resize到448，剩余的边pad到能被32整除
    '''
    img_metas context
    'filename': 'data/casia-SPT_val/val/JPEGImages/00238.jpg', 
    'ori_shape': (402, 600, 3), 'img_shape': (448, 669, 3), 
    'pad_shape': (448, 672, 3), 'scale_factor': 1.1144278606965174, 'flip': False, 
    'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 
    'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}}

    '''

    def forward_test(self, imgs, img_metas, **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_meta (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(imgs), len(img_metas)))
        # TODO: remove the restriction of imgs_per_gpu == 1 when prepared
        imgs_per_gpu = imgs[0].size(0)
        assert imgs_per_gpu == 1

        if num_augs == 1:
            return self.simple_test(imgs[0], img_metas[0], **kwargs)
        else:
            return self.aug_test(imgs, img_metas, **kwargs)
    
    def simple_test(self, img, img_meta, rescale=False):
       
        #test_tensor = torch.ones(1,3,448,512).cuda()
        #x = self.extract_feat(test_tensor)
        x = self.extract_feat(img)

        outs = self.bbox_head(x,eval=True)
  
        mask_feat_pred = self.mask_feat_head(
                x[self.mask_feat_head.
                  start_level:self.mask_feat_head.end_level + 1])

        seg_inputs = outs + (mask_feat_pred, img_meta, self.test_cfg, rescale)

        seg_result = self.bbox_head.get_seg(*seg_inputs)
        return seg_result

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation."""
        raise NotImplementedError


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