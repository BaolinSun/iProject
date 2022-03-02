'''
Author: sunbaolin
Date: 2022-02-07 18:10:31
LastEditors: sunbaolin
LastEditTime: 2022-03-03 10:53:01
Description: file content
FilePath: /iProject/train.py
'''

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import os
import argparse
import torch.distributed as dist
import pynvml

from torch.utils.data import DataLoader
from config import cfg
from functools import partial
from datasets.coco import CocoDataset
from datasets.collate import collate
from datasets.sampler import GroupSampler, DistributedGroupSampler
from models.ins_his import INS_HIS
from utils import gradinator, set_lr, get_warmup_lr, cal_time, bcolors
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, rank, world_size) -> None:
        self.parse_args()
        self.init_distributed(rank, world_size)
        self.init_datasets()
        self.init_model()
        self.train()
        self.cleanup()

    def parse_args(self):
        parser = argparse.ArgumentParser()
        # Training setting
        parser.add_argument('--batch-size-per-gpu', type=int, default=4)
        parser.add_argument('--num-workers', type=int, default=8)
        parser.add_argument('--epoch-start', type=int, default=0)
        parser.add_argument('--epoch-end', type=int, default=144)
        # Distributed
        parser.add_argument('--distributed-addr', type=str, default='localhost')
        parser.add_argument('--distributed-port', type=str, default='12355')
        self.args = parser.parse_args()
        print(self.args)

    def init_distributed(self, rank, world_size):
        print('Initializing distributed')
        self.rank = rank
        self.world_size = world_size
        os.environ['MASTER_ADDR'] = self.args.distributed_addr
        os.environ['MASTER_PORT'] = self.args.distributed_port
        dist.init_process_group(backend='nccl', init_method='env://', world_size=self.world_size, rank=self.rank) 

        pynvml.nvmlInit()

    def init_datasets(self):

        self.log('Initializing datasets')
        # datasets
        self.cocodata = CocoDataset(ann_file=cfg.dataset.train_info,
                               img_prefix=cfg.dataset.trainimg_prefix,
                               data_root=cfg.dataset.train_prefix)

        # sampler
        self.sampler = DistributedGroupSampler(self.cocodata, self.args.batch_size_per_gpu)

        # dataloader
        self.cocoloader = DataLoader(dataset=self.cocodata,
                                     batch_size=self.args.batch_size_per_gpu,
                                     sampler=self.sampler,
                                     num_workers=self.args.num_workers,
                                     collate_fn=partial(collate, samples_per_gpu=self.args.batch_size_per_gpu),
                                     pin_memory=False)

    def init_model(self):
        self.log('Initializing model')

        torch.manual_seed(0)
        torch.cuda.set_device(self.rank)
        
        # model
        if cfg.pretrained is None:
            model = INS_HIS(cfg, pretrained=None, mode='train')
        else:
            model = INS_HIS(cfg, pretrained=cfg.pretrained, mode='test')

        model = model.cuda(self.rank)
        model = model.train()
        self.model = nn.parallel.DistributedDataParallel(model, device_ids=[self.rank])

        # optimizer
        self.optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001)

    def init_writer(self):
        if self.rank == 0:
            self.log('Initializing writer')
            self.writer = SummaryWriter('runs/log')

    def train(self):
        epoch_size = int(len(self.cocodata) / self.args.batch_size_per_gpu / self.world_size)
        base_nums = self.args.epoch_start - 1
        for epoch in range(self.args.epoch_start, self.args.epoch_end):

            start = time.time()

            if epoch < cfg.lr_config['step'][0]:
                set_lr(self.optimizer, 0.01)
                base_lr = 0.01
                cur_lr = 0.01
            elif epoch >= cfg.lr_config['step'][
                    0] and epoch < cfg.lr_config['step'][1]:
                set_lr(self.optimizer, 0.001)
                base_lr = 0.001
                cur_lr = 0.001
            elif epoch >= cfg.lr_config['step'][
                    1] and epoch < cfg.lr_config['step'][2]:
                set_lr(self.optimizer, 0.0001)
                base_lr = 0.0001
                cur_lr = 0.0001
            elif epoch >= cfg.lr_config['step'][
                    2] and epoch <= cfg.epoch_iters_total:
                set_lr(self.optimizer, 0.00001)
                base_lr = 0.00001
                cur_lr = 0.00001
            else:
                raise NotImplementedError("train epoch is done!")

            loss_sum = 0
            loss_ins = 0
            loss_cate = 0

            for i, data in enumerate(self.cocoloader):

                if cfg.lr_config[
                        'warmup'] is not None and base_nums < cfg.lr_config[
                            'warmup_iters']:
                    warm_lr = get_warmup_lr(base_nums,
                                            cfg.lr_config['warmup_iters'],
                                            cfg.lr_config['base_lr'],
                                            cfg.lr_config['warmup_ratio'],
                                            cfg.lr_config['warmup'])
                    set_lr(self.optimizer, warm_lr)
                    cur_lr = warm_lr
                else:
                    set_lr(self.optimizer, base_lr)
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

                loss = self.model.forward(img=imgs,
                                    img_meta=img_meta,
                                    gt_bboxes=gt_bboxes,
                                    gt_labels=gt_labels,
                                    gt_masks=gt_masks)
                losses = loss['loss_ins'] + loss['loss_cate']

                self.optimizer.zero_grad()
                losses.backward()
                if torch.isfinite(losses).item():
                    self.optimizer.step()
                else:
                    NotImplementedError("loss type error!can't backward!")

                loss_sum = loss_sum + losses.cpu().item()
                loss_ins = loss_ins + loss['loss_ins'].cpu().item()
                loss_cate = loss_cate + loss['loss_cate'].cpu().item()

                if i % cfg.train_show_interval == 0:

                    out_srt = 'epoch: [' + str(epoch) + '/' + str(cfg.epoch_iters_total) + '] ' + \
                            '[' + str(i) + '/' + str(epoch_size) + ']'
                    out_srt = out_srt + ' loss: {:6.4f}'.format(
                        loss_sum / float(cfg.train_show_interval))
                    out_srt = out_srt + ' loss_ins: {:6.4f}'.format(
                        loss_ins / float(cfg.train_show_interval))
                    out_srt = out_srt + ' loss_cate: {:6.4f}'.format(
                        loss_cate / float(cfg.train_show_interval))
                    out_srt = out_srt + ' lr: {:7.5f}'.format(cur_lr)
                    self.log(out_srt)
                    loss_sum = 0.0
                    loss_ins = 0.0
                    loss_cate = 0.0
    
            finish = time.time()
            self.log('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

            self.save(epoch)

    def save(self, epoch):

        if self.rank == 1:
            save_name = "./checkpoints/runtime/model_" + cfg.backbone.name + "_epoch_" + str(epoch) + ".pth"
            self.log(bcolors.HEADER + save_name + bcolors.ENDC)
            self.model.module.save_weights(save_name)
            self.log('Model saved')
            icmd = 'python inference.py --model-path checkpoints/runtime/model_' + cfg.backbone.name + '_epoch_{}.pth --input-source data/coco/val2022 --eval'.format(epoch)
            self.log(icmd)
            os.system(icmd)

    def cleanup(self):
        dist.destroy_process_group()

    def log(self, msg):
        print(bcolors.BLUE + f'[GPU{self.rank}] ' + bcolors.ENDC + f'{msg}')

import time
def test(rank, world_size):
    pynvml.nvmlInit()
    meminfo = [0] * world_size

    while True:
        meminfo[rank] = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(rank)).free
        print('rank:', rank,meminfo)
        time.sleep(1)

import torch.multiprocessing as mp

if __name__ == '__main__':
    world_size = torch.cuda.device_count()
    mp.spawn(Trainer, nprocs=world_size, args=(world_size,), join=True)
