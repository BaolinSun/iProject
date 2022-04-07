'''
Author: sunbaolin
Date: 2022-02-15 09:37:53
LastEditors: sunbaolin
LastEditTime: 2022-02-16 21:36:11
FilePath: /iProject/inference.py
Description: 

'''


import torch
import time
import argparse
import numpy as np

from glob import glob
from tqdm import tqdm
from config import cfg
from config import resnet34_backbone
from config import kernel_head_light
from config import fpn_light
from compose import Compose
from utils import result2json, result2image, result2mask, run_eval_miou
from models.ins_his import INS_HIS
from eval_mask import run_eval_mask
from interact_segm import interact_segm
from datasets.piplines import Resize, Normalize, Pad, ImageToTensor, TestCollect, MultiScaleFlipAug

class LoadImage(object):
    def __init__(self, resolution) -> None:
        self.width = resolution[0]
        self.height = resolution[1]

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None 

        img = np.empty((self.height, self.width, 3))
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


test_process_pipelines = [
    Resize(keep_ratio=True),
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    Pad(size_divisor=32),
    ImageToTensor(keys=['img']),
    TestCollect(keys=['img']),
]
Multest = MultiScaleFlipAug(transforms=test_process_pipelines, img_scale=(1333, 800), flip=False)

def speed_test(model_path, resolution):    
    test_pipeline = []
    test_pipeline.append(LoadImage(resolution))
    test_pipeline.append(Multest)
    test_pipeline = Compose(test_pipeline)

    cfg.backbone = resnet34_backbone
    cfg.kernel_head = kernel_head_light
    cfg.fpn = fpn_light
    
    model = INS_HIS(cfg, pretrained=model_path, mode='test')
    model = model.cuda()

    start_time = time.time()
    count = 0
    for i in tqdm(range(100)):

        data = dict(img='tmp')
        data = test_pipeline(data)
        imgs = data['img']
        img = imgs[0].cuda().unsqueeze(0)
        img_info = data['img_metas']

        with torch.no_grad():
            seg_result = model.forward(img=[img], img_meta=[img_info], return_loss=False)

        count += 1

    end_time = time.time()

    fps = (count / (end_time - start_time))

    print('....')
    print('FPS on resolution[{}x{}]: {}'.format(resolution[0], resolution[1], fps))
    print('....')
    print('finshed...')

    return fps


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None)
    parser.add_argument('--resolution', type=int, required=True, nargs=2)
    args = parser.parse_args()
    print(args)

    fps = speed_test(model_path = args.model, resolution=args.resolution)
