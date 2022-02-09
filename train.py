'''
Author: sunbaolin
Date: 2022-02-07 18:10:31
LastEditors: Please set LastEditors
LastEditTime: 2022-02-09 17:23:32
Description: file content
FilePath: /iProject/train.py
'''

import torch
import torch.nn as nn
import numpy as np

from torch.utils.data import DataLoader
from config import cfg
from datasets.coco import CocoDataset
from functools import partial

from pprint import pprint


def train():

    # process config
    batch_size = cfg.imgs_per_gpu * cfg.num_gpus
    num_workers = cfg.workers_per_gpu * cfg.num_gpus

    # datasets
    cocodata = CocoDataset(ann_file=cfg.dataset.train_info,
                           img_prefix=cfg.dataset.trainimg_prefix,
                           data_root=cfg.dataset.train_prefix)

    # dataloader
    cocoloader = DataLoader(dataset=cocodata,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=False)

    print('batch size:', batch_size)


    for i, data in enumerate(cocoloader):
        print(i)
        break

if __name__ == '__main__':
    train()
    print('sccess...')
