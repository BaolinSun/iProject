'''
Author: sunbaolin
Date: 2022-02-15 09:37:53
LastEditors: sunbaolin
LastEditTime: 2022-02-16 21:36:11
FilePath: /iProject/inference.py
Description: 

'''


import torch
import torch.nn as nn
import os
import shutil
import argparse
import json

from glob import glob
from tqdm import tqdm
from config import cfg
from compose import Compose
from utils import result2json
from models.ins_his import INS_HIS
from eval_mask import run_eval_mask
from datasets.piplines import LoadImage, Resize, Normalize, Pad, ImageToTensor, TestCollect, MultiScaleFlipAug

test_process_pipelines = [
    Resize(keep_ratio=True),
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    Pad(size_divisor=32),
    ImageToTensor(keys=['img']),
    TestCollect(keys=['img']),
]
Multest = MultiScaleFlipAug(transforms=test_process_pipelines, img_scale = (480, 448), flip=False)
test_pipeline = []
test_pipeline.append(LoadImage())
test_pipeline.append(Multest)
test_pipeline = Compose(test_pipeline)

def inference(model_path, input_source, output_source = None, eval_flag=False, test_mode='images'):
    
    model = INS_HIS(cfg, pretrained=model_path, mode='test')
    model = model.cuda()

    if output_source != None:
        if os.path.exists(output_source):
            shutil.rmtree(output_source)
        os.makedirs(output_source)

    results = []
    images = glob(input_source+'/*')
    for imgpath in tqdm(images):
        pathname,filename = os.path.split(imgpath)
        prefix,suffix = os.path.splitext(filename)
        img_id = int(prefix)
        img_id = str(img_id)

        data = dict(img=imgpath)
        data = test_pipeline(data)
        imgs = data['img']
        img = imgs[0].cuda().unsqueeze(0)
        img_info = data['img_metas']

        with torch.no_grad():
            seg_result = model.forward(img=[img], img_meta=[img_info], return_loss=False)

        if eval_flag == True:
            try:
                result = result2json(img_id, seg_result)
                results.append(result)
            except:
                pass

    if eval_flag == True:
        re_js = json.dumps(results)
        fjson = open("eval_masks.json","w")
        fjson.write(re_js)
        fjson.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-path', type=str, default=cfg.model_path)
    parser.add_argument('--input-source', type=str, default=cfg.input_source)
    parser.add_argument('--output-source', type=str, default=cfg.output_source)
    parser.add_argument('--eval', action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args.model_path, args.input_source, args.output_source, args.eval)

    inference(model_path = args.model_path,
            input_source = args.input_source,
            output_source = args.output_source,
            eval_flag = args.eval)
    print('success...')

    if args.eval:
        run_eval_mask(
            os.path.join(cfg.dataset.valid_prefix, cfg.dataset.valid_info),
            'eval_masks.json')
