'''
Author: sunbaolin
Contact: baolin.sun@mail.sdu.edu.cn
Date: 2022-02-07 18:43:09
LastEditors: sunbaolin
LastEditTime: 2022-03-03 08:49:18
Description: file content
FilePath: /iProject/utils.py
'''

import os
import cv2
import numpy as np
import pandas as pd
import pycocotools.mask as maskutil

from scipy import ndimage
from datasets.piplines import imresize
from tqdm import tqdm
from tqdm.contrib import tzip

class bcolors:
    HEADER = '\033[95m'
    BLUE = '\033[1;94m'
    GREEN = '\033[1;92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


def cal_time(timestamp):
    tm_min = timestamp / 60.0
    tm_hour = tm_min / 60
    # tm_mday = tm_min // 24
    # tm_hour = tm_hour % 24

    return tm_hour


# set learning rate
def set_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr


def gradinator(x):
    x.requires_grad = False
    return x


def get_warmup_lr(cur_iters,
                  warmup_iters,
                  bash_lr,
                  warmup_ratio,
                  warmup='linear'):
    if warmup == 'constant':
        warmup_lr = bash_lr * warmup_ratio
    elif warmup == 'linear':
        k = (1 - cur_iters / warmup_iters) * (1 - warmup_ratio)
        warmup_lr = bash_lr * (1 - k)
    elif warmup == 'exp':
        k = warmup_ratio**(1 - cur_iters / warmup_iters)
        warmup_lr = bash_lr * k
    return warmup_lr


class Config(object):
    """
    After implement this class, you can call 'cfg.x' instead of 'cfg['x']' to get a certain parameter.
    """

    def __init__(self, config_dict):
        for key, val in config_dict.items():
            self.__setattr__(key, val)

    def copy(self, new_config_dict={}):
        """
        Copies this config into a new config object, making the changes given by new_config_dict.
        """
        ret = Config(vars(self))
        for key, val in new_config_dict.items():
            ret.__setattr__(key, val)

        return ret

    def replace(self, new_config_dict):
        """
        Copies new_config_dict into this config object. Note: new_config_dict can also be a config object.
        """
        if isinstance(new_config_dict, Config):
            new_config_dict = vars(new_config_dict)

        for key, val in new_config_dict.items():
            self.__setattr__(key, val)

    def __repr__(self):
        return self.name

    def print(self):
        for k, v in vars(self).items():
            print(k, ' = ', v)


COCO_LABEL = [1]
COCO_CLASSES = ('person', )
def result2json(img_id, result):
    rel = []
    seg_pred = result[0][0].cpu().numpy().astype(np.uint8)
    cate_label = result[0][1].cpu().numpy().astype(np.int)
    cate_score = result[0][2].cpu().numpy().astype(np.float)
    num_ins = seg_pred.shape[0]
    for j in range(num_ins):
        realclass = COCO_LABEL[cate_label[j]]
        re = {}
        score = cate_score[j]
        re["image_id"] = img_id
        re["category_id"] = int(realclass)
        re["score"] = float(score)
        if re["score"] < 0.3:
            continue
        outmask = np.squeeze(seg_pred[j])
        outmask = outmask.astype(np.uint8)
        outmask=np.asfortranarray(outmask)
        rle = maskutil.encode(outmask)
        rle['counts'] = rle['counts'].decode('ascii')
        re["segmentation"] = rle
        rel.append(re)
    return rel


def result2image(img, result, score_thr=0.3):
    if isinstance(img, str):
        img = cv2.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape

    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_index = score > score_thr
    seg_label = seg_label[vis_index]
    cate_label = cate_label[vis_index]
    cate_score = score[vis_index]
    num_mask = seg_label.shape[0]

    np.random.seed(512)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]

    for idx in range(num_mask):
        # idx = -(idx+1)
        if cate_label[idx] == 0:
            cur_mask = seg_label[idx]
            cur_mask = imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            color_mask = color_masks[idx]
            cur_mask_bool = cur_mask.astype(np.bool)
            img_show[cur_mask_bool] = img[cur_mask_bool] * 0.5 + color_mask * 0.5

            cur_cate = cate_label[idx]
            cur_score = cate_score[idx]

            label_text = COCO_CLASSES[cur_cate]
            label_text += '={:.02f}'.format(cur_score)
            center_y, center_x = ndimage.measurements.center_of_mass(cur_mask)
            vis_pos = (int(center_x), int(center_y))
            cv2.putText(img_show, label_text, vis_pos, cv2.FONT_HERSHEY_DUPLEX, 0.3, (255, 255, 255))

    return img_show


def result2mask(img, result, score_thr=0.3):
    if isinstance(img, str):
        img = cv2.imread(img)
    img_show = img.copy()
    h, w, _ = img.shape
    img_show = np.zeros_like(img)

    cur_result = result[0]
    seg_label = cur_result[0]
    seg_label = seg_label.cpu().numpy().astype(np.uint8)
    cate_label = cur_result[1]
    cate_label = cate_label.cpu().numpy()
    score = cur_result[2].cpu().numpy()

    vis_index = score > score_thr
    seg_label = seg_label[vis_index]
    cate_label = cate_label[vis_index]
    cate_score = score[vis_index]
    num_mask = seg_label.shape[0]

    np.random.seed(512)
    color_masks = [
        np.random.randint(0, 256, (1, 3), dtype=np.uint8)
        for _ in range(num_mask)
    ]

    for idx in range(num_mask):
        # idx = -(idx+1)
        if cate_label[idx] == 0:
            cur_mask = seg_label[idx]
            cur_mask = imresize(cur_mask, (w, h))
            cur_mask = (cur_mask > 0.5).astype(np.uint8)
            if cur_mask.sum() == 0:
                continue
            color_mask = color_masks[idx]
            cur_mask_bool = cur_mask.astype(np.bool)
            img_show[cur_mask_bool] = 255

    return img_show

def run_eval_miou(pha_file, mask_file):
    pha_list = os.listdir(pha_file)
    mask_list = os.listdir(mask_file)
    pha_list.sort()
    mask_list.sort()

    iou = []
    for fmask in tqdm(mask_list):
        pha = cv2.imread(os.path.join(pha_file, fmask))
        mask = cv2.imread(os.path.join(mask_file, fmask))

        intersection = np.sum(np.logical_and(mask, pha))
        union = np.sum(np.logical_or(mask, pha))

        if (union == 0) or (intersection == 0):
            continue
        iou_score = intersection / union
        iou.append(iou_score)

    iou = pd.DataFrame(columns = ['iou'], data = iou)
    print(iou)
    print('....  ...')
    print('miou:', iou['iou'].mean())
    print('....  ...')