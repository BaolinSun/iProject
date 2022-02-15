import json
import os
import os.path as osp
from pycocotools.coco import COCO
import pycocotools.mask as maskUtils

import numpy as np
from pprint import pprint
import cv2
import pandas as pd

from tqdm import tqdm
from config import cfg

import logging

logging.basicConfig(level=logging.WARNING,#控制台打印的日志级别
                    filename='miou.log',
                    filemode='a',##模式，有w和a，w就是写模式，每次都会重新写日志，覆盖之前的日志
                    #a是追加模式，默认如果不写的话，就是追加模式
                    format='%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
                    #日志格式
                    )

def run_eval_mask(ann_file, mak_file):

    # ann_file = '/home/shuyu/Datasets/coco/annotations/instances_val2017.json'
    # mak_file = '/home/shuyu/workspace/insseg_hisense/eval_masks.json'

    coco = COCO(ann_file)

    def coco_mask(id):

        cat_ids = coco.getCatIds()
        cat2label = {
            cat_id: i+1
            for i, cat_id in enumerate(cat_ids)
        }

        img_ids = coco.getImgIds()
        img_infos = []
        for i in img_ids:
            info = coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)

        img_id = img_infos[id]['id']
        ann_ids = coco.getAnnIds(imgIds=[img_id])
        ann_infos = coco.loadAnns(ann_ids)
        img_info = img_infos[id]

        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_infos):
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
                gt_labels.append(cat2label[ann['category_id']])
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

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        results = dict(img_info=img_info, ann_info=ann)
        results['img_prefix'] = []
        results['seg_prefix'] = []
        results['proposal_file'] = []
        results['bbox_fields'] = []
        results['mask_fields'] = []
        results['seg_fields'] = []

        def load_bboxes(results):
            ann_info = results['ann_info']
            results['gt_bboxes'] = ann_info['bboxes']

            gt_bboxes_ignore = ann_info.get('bboxes_ignore', None)
            if gt_bboxes_ignore is not None:
                results['gt_bboxes_ignore'] = gt_bboxes_ignore
                results['bbox_fields'].append('gt_bboxes_ignore')
            results['bbox_fields'].append('gt_bboxes')
            return results

        results = load_bboxes(results)

        def load_labels(results):
            results['gt_labels'] = results['ann_info']['labels']
            return results

        results = load_labels(results)

        def poly2mask(mask_ann, img_h, img_w):
            if isinstance(mask_ann, list):
                rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
                rle = maskUtils.merge(rles)
            elif isinstance(mask_ann['counts'], list):
                # uncompressed RLE
                rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            else:
                # rle
                rle = mask_ann

            mask = maskUtils.decode(rle)

            return mask

        def load_masks(results):
            h, w = results['img_info']['height'], results['img_info']['width']
            gt_masks = results['ann_info']['masks']
            gt_masks = [poly2mask(mask, h, w) for mask in gt_masks]
            results['gt_masks'] = gt_masks
            results['mask_fields'].append('gt_masks')

            return results

        results = load_masks(results)

        return results

    # COCO_MASK = [None for _ in range(coco.getImgIds())]

    ts = []
    gt_mask = {}
    for i in tqdm(range(len(coco.getImgIds()))):
        results = coco_mask(i)
        # COCO_MASK.append(results)
        gt = np.zeros((results['img_info']['height'], results['img_info']['width']))
        gt_id = results['img_info']['id']
        human_flag = 0
        for gt_index, gt_labels in enumerate(results['gt_labels']):
            gt = gt + results['gt_masks'][gt_index]
            human_flag = 1

        if human_flag == 1:
            gt_mask[gt_id] = gt


    with open(mak_file, 'r') as f:
        masks = json.load(f)

    dt_mask = {}
    for item in tqdm(masks):
        if len(item) == 0:
            continue

        dt = np.zeros((item[0]['segmentation']['size']))

        dt_id = item[0]['image_id']
                
        for mask in item:
            # if (mask['score'] > 0.5) and (mask['category_id'] == 1):
            dt = dt + maskUtils.decode(mask['segmentation'])

        dt_mask[int(dt_id)] = dt


    dt_keys = dt_mask.keys()
    gt_keys = gt_mask.keys()

    iou = []
    img_files = []
    for keys in tqdm(dt_keys):
        try:
            dt = dt_mask[keys]
            gt = gt_mask[keys]

            intersection = np.sum(np.logical_and(dt, gt))
            union = np.sum(np.logical_or(dt, gt))

            if (union == 0) or (intersection == 0):
                continue
            iou_score = intersection / union
            iou.append(iou_score)
        except:
            # print(type(keys), keys)
            pass

    iou = pd.DataFrame(columns = ['iou'], data = iou)
    print(iou)
    print('....  ...')
    print('miou:', iou['iou'].mean())
    logging.warning(iou['iou'].mean())
    print('....  ...')


if __name__ == '__main__':
    run_eval_mask(
            os.path.join(cfg.dataset.valid_prefix, cfg.dataset.valid_info),
            'eval_masks.json')
# name = ['file_name']
# img_files = pd.DataFrame(columns = name, data = img_files)
# img_files.to_csv('img_file.csv')



# img = results['gt_masks'][1] * 255
# img = maskUtils.decode(masks[1000]['segmentation'])*255
# cv2.imshow('img', img)
# cv2.waitKey()

# print(type(grle[0]['counts']))
# iou = maskUtils.iou(grle[0], grle[1], [0, 0])

# from pycocotools._mask import RLEs, _frString


# dt = _frString(maskUtils.encode(results['gt_masks'][0]))
# gt = _frString(maskUtils.encode(results['gt_masks'][1]))

# dt = results['gt_masks'][0]
# gt = results['gt_masks'][1]

# a = np.logical_or(dt, gt)
# print(a.sum())