'''
Author: sunbaolin
Contact: baolin.sun@mail.sdu.edu.cn
Date: 2022-02-07 18:43:09
LastEditors: sunbaolin
LastEditTime: 2022-03-03 08:49:18
Description: file content
FilePath: /iProject/utils.py
'''

import numpy as np
import pycocotools.mask as maskutil

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
        outmask = np.squeeze(seg_pred[j])
        outmask = outmask.astype(np.uint8)
        outmask=np.asfortranarray(outmask)
        rle = maskutil.encode(outmask)
        rle['counts'] = rle['counts'].decode('ascii')
        re["segmentation"] = rle
        rel.append(re)
    return rel