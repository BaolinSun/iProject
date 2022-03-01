'''
Author: sunbaolin
Date: 2022-02-07 18:10:31
LastEditors: sunbaolin
LastEditTime: 2022-02-28 20:42:13
Description: file content
FilePath: /iProject/train.py
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os

from torch.utils.data import DataLoader
from config import cfg
from functools import partial
from datasets.coco import CocoDataset
from datasets.collate import collate
from datasets.sampler import GroupSampler
from models.ins_his import INS_HIS
from utils import gradinator, set_lr, get_warmup_lr, cal_time, bcolors
from torch.utils.tensorboard import SummaryWriter


def train():

    # process config
    batch_size = cfg.imgs_per_gpu * cfg.num_gpus
    num_workers = cfg.workers_per_gpu * cfg.num_gpus

    # datasets
    cocodata = CocoDataset(ann_file=cfg.dataset.train_info,
                           img_prefix=cfg.dataset.trainimg_prefix,
                           data_root=cfg.dataset.train_prefix)

    # sampler
    sampler = GroupSampler(cocodata, cfg.imgs_per_gpu)

    # dataloader
    cocoloader = DataLoader(dataset=cocodata,
                            batch_size=batch_size,
                            sampler=sampler,
                            num_workers=num_workers,
                            collate_fn=partial(collate, samples_per_gpu=cfg.imgs_per_gpu),
                            pin_memory=False)

    # model
    if cfg.pretrained is None:
        model = INS_HIS(cfg, pretrained=None, mode='train')
    else:
        model = INS_HIS(cfg, pretrained=cfg.pretrained, mode='test')
    model = model.cuda()
    model = model.train()

    # optimizer
    optimizer = optim.SGD(model.parameters(),
                          lr=0.01,
                          momentum=0.9,
                          weight_decay=0.0001)

    # process config
    epoch_size = len(cocodata) // batch_size
    base_loop = cfg.epoch_iters_start
    base_nums = (base_loop-1) * epoch_size
    total_nums = (cfg.epoch_iters_total - cfg.epoch_iters_start + 1) * epoch_size
    cur_nums = 0

    writer = SummaryWriter('runs/log')
    writer_index = 0
                          

    for iter_nums in range(cfg.epoch_iters_start, cfg.epoch_iters_total+1):

        # learning rate
        if iter_nums < cfg.lr_config['step'][0]:
            set_lr(optimizer, 0.01)
            base_lr = 0.01
            cur_lr = 0.01
        elif iter_nums >= cfg.lr_config['step'][0] and iter_nums < cfg.lr_config['step'][1]:
            set_lr(optimizer, 0.001)
            base_lr = 0.001
            cur_lr = 0.001
        elif iter_nums >= cfg.lr_config['step'][1] and iter_nums < cfg.lr_config['step'][2]:
            set_lr(optimizer, 0.0001)
            base_lr = 0.0001
            cur_lr = 0.0001
        elif iter_nums >= cfg.lr_config['step'][2] and iter_nums <= cfg.epoch_iters_total:
            set_lr(optimizer, 0.00001)
            base_lr = 0.00001
            cur_lr = 0.00001
        else:
            raise NotImplementedError("train epoch is done!")

        loss_sum = 0
        loss_ins = 0
        loss_cate = 0

        for i, data in enumerate(cocoloader):

            if cfg.lr_config['warmup'] is not None and base_nums < cfg.lr_config['warmup_iters']:
                warm_lr = get_warmup_lr(base_nums, cfg.lr_config['warmup_iters'], cfg.lr_config['base_lr'], cfg.lr_config['warmup_ratio'], cfg.lr_config['warmup'])
                set_lr(optimizer, warm_lr)
                cur_lr = warm_lr
            else:
                set_lr(optimizer, base_lr)
                cur_lr = base_lr

            last_time = time.time()

            # image
            imgs = gradinator(data['img'].data[0].cuda())
            img_meta = data['img_metas'].data[0]

            # bbox
            gt_bboxes = []
            for bbox in data['gt_bboxes'].data[0]:
                bbox = gradinator(bbox.cuda())
                gt_bboxes.append(bbox)

            # mask
            gt_masks = data['gt_masks'].data[0]

            # label
            gt_labels = []
            for label in data['gt_labels'].data[0]:
                label = gradinator(label.cuda())
                gt_labels.append(label)

            loss = model.forward(img=imgs,
                                 img_meta=img_meta,
                                 gt_bboxes=gt_bboxes,
                                 gt_labels=gt_labels,
                                 gt_masks=gt_masks)
            losses = loss['loss_ins'] + loss['loss_cate']

            optimizer.zero_grad()
            losses.backward()
            if torch.isfinite(losses).item():
                optimizer.step()
            else:
                NotImplementedError("loss type error!can't backward!")

            use_time = time.time() - last_time
            base_nums = base_nums + 1
            cur_nums = cur_nums + 1

            loss_sum = loss_sum + losses.cpu().item()
            loss_ins = loss_ins + loss['loss_ins'].cpu().item()
            loss_cate = loss_cate + loss['loss_cate'].cpu().item()

            writer.add_scalar('loss', losses.cpu().item(), global_step=writer_index)
            writer.add_scalar('loss_ins', loss['loss_ins'].cpu().item(), global_step=writer_index)
            writer.add_scalar('loss_cate', loss['loss_cate'].cpu().item(), global_step=writer_index)
            writer_index += 1

            if i % cfg.train_show_interval == 0:

                left_time = use_time * (total_nums - cur_nums)
                left_time = cal_time(left_time)

                out_srt = bcolors.OKBLUE + \
                        'epoch:[' + str(iter_nums) + ']/[' + str(cfg.epoch_iters_total) + '] ' + \
                        '[' + str(i) + ']/[' + str(epoch_size) + ']' + \
                        bcolors.ENDC
                out_srt = out_srt +  ' left_time: {:5.1f}'.format(left_time) + 'h,'
                out_srt = out_srt + ' loss: {:6.4f}'.format(loss_sum / float(cfg.train_show_interval))
                out_srt = out_srt + ' loss_ins: {:6.4f}'.format(loss_ins / float(cfg.train_show_interval))
                out_srt = out_srt + ' loss_cate: {:6.4f}'.format(loss_cate / float(cfg.train_show_interval))
                out_srt = out_srt + ' lr: {:7.5f}'.format(cur_lr)
                print(out_srt)
                loss_sum = 0.0
                loss_ins = 0.0
                loss_cate = 0.0

        save_name = "./checkpoints/runtime/model_" + cfg.backbone.name + "_epoch_" + str(iter_nums) + ".pth"
        print(bcolors.HEADER + save_name + bcolors.ENDC)
        model.save_weights(save_name)

        icmd = 'python inference.py --model-path checkpoints/runtime/model_' + cfg.backbone.name + '_epoch_{}.pth --input-source data/coco/val2022 --eval'.format(iter_nums)
        print(icmd)
        os.system(icmd)


if __name__ == '__main__':
    train()
    print('sccess...')
