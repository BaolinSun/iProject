'''
Author: sunbaolin
Date: 2022-02-07 18:28:49
LastEditors: sunbaolin
LastEditTime: 2022-02-10 19:32:21
Description: file content
FilePath: /iProject/datasets/coco.py
'''

import torch
import torch.nn as nn
import torchvision
import numpy as np
import os

from torch.utils.data import Dataset
# from torchvision.transforms import Compose
from compose import Compose
from pycocotools.coco import COCO
from .piplines import LoadImageFromFile, LoadAnnotations, Resize, RandomFlip, Normalize, Pad, DefaultFormatBundle, Collect



# train
train_process_pipelines = [
    LoadImageFromFile(),
    LoadAnnotations(with_bbox=True, with_mask=True),
    Resize(img_scale=[(768, 512), (768, 480), (768, 448), (768, 416), (768, 384), (768, 352)], multiscale_mode='value', keep_ratio=True),
    RandomFlip(flip_ratio=0.5),
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    Pad(size_divisor=32),
    DefaultFormatBundle(),
    Collect(keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'], meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'img_norm_cfg'))
]

class CocoDataset(Dataset):

    def __init__(self,
                 ann_file,
                 pipeline=None,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 test_mode=False,
                 filter_empty_gt=True) -> None:
        super().__init__()
        self.ann_file = ann_file
        self.data_root = data_root
        self.img_prefix = img_prefix
        self.seg_prefix = seg_prefix
        self.test_mode = test_mode
        self.proposal_file = None
        self.filter_empty_gt = filter_empty_gt

        self.ann_file = os.path.join(self.data_root, self.ann_file)
        self.img_prefix = os.path.join(self.data_root, self.img_prefix)

        self.img_infos = self.load_annotations(self.ann_file)
        '''TODO'''
        self.proposals = None

        # filter images too small or without ground truths.
        if not test_mode:
            valid_inds = self._filter_imgs()
            self.img_infos = [self.img_infos[i] for i in valid_inds]
            if self.proposals is not None:
                self.proposals = [self.proposals[i] for i in valid_inds]

        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()

        # process pipeline
        self.transform = Compose(train_process_pipelines)

        print('build datasets...')

    def __len__(self):
        return len(self.img_infos)

    def __getitem__(self, index):
        if self.test_mode:
            raise NotImplementedError
        while True:
            data = self.prepare_train_img(index)
            if data is None:
                index = self._rand_another(index)
                continue
            return data

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.filter_empty_gt and self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)

        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos[i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        if self.proposals is not None:
            results['proposals'] = self.proposals[idx]

        self.pre_pipeline(results)
        return self.transform(results)

    def pre_pipeline(self, results):
        results['img_prefix'] = self.img_prefix
        results['seg_prefix'] = self.seg_prefix
        results['proposal_file'] = self.proposal_file
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(bboxes=gt_bboxes,
                   labels=gt_labels,
                   bboxes_ignore=gt_bboxes_ignore,
                   masks=gt_masks_ann,
                   seg_map=seg_map)

        return ann