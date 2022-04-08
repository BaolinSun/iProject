'''
Author: sunbaolin
Date: 2022-02-15 09:37:53
LastEditors: sunbaolin
LastEditTime: 2022-02-16 21:36:11
FilePath: /iProject/inference.py
Description: 

'''


import sys
import torch
import torch.nn as nn
import os
import shutil
import argparse
import json
import cv2

from glob import glob
from tqdm import tqdm
from config import cfg
from compose import Compose
from utils import result2json, result2image, result2mask, run_eval_miou, run_eval_miou_simple
from utils import prepare_images, prepare_output
from models.ins_his import INS_HIS
from eval_mask import run_eval_mask
from speed_test import speed_test
from interact_segm import interact_segm
from datasets.piplines import LoadImage, Resize, Normalize, Pad, ImageToTensor, TestCollect, MultiScaleFlipAug

test_process_pipelines = [
    Resize(keep_ratio=True),
    Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
    Pad(size_divisor=32),
    ImageToTensor(keys=['img']),
    TestCollect(keys=['img']),
]
Multest = MultiScaleFlipAug(transforms=test_process_pipelines, img_scale=(1333, 800), flip=False)
test_pipeline = []
test_pipeline.append(LoadImage())
test_pipeline.append(Multest)
test_pipeline = Compose(test_pipeline)

def inference(model_path, input_source, output_source=None, out_flag = False, eval_flag=False, json_flag=False, mask_flag=False, test_mode='images'):
    
    print('load model...')
    model = INS_HIS(cfg, pretrained=model_path, mode='test')
    model = model.cuda()

    if output_source != None:
        if os.path.exists(output_source):
            shutil.rmtree(output_source)
        os.makedirs(output_source)
    # if output_source != None:
    #     if os.path.exists(output_source):
    #         shutil.rmtree(output_source)
    #     os.makedirs(output_source)
    # if eval_flag:
    #     mask_file = cfg.mask_output
    #     if os.path.exists(mask_file):
    #         shutil.rmtree(mask_file)
    #     os.makedirs(mask_file)

    if eval_flag or out_flag:
        print('prepare mask output...')
        prepare_output(input_source, cfg)
        print('prepare eval images...')
        images = prepare_images(input_source)
    else:
        images = glob(input_source+'/*')

    results = []
    print('inference...')
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

        if eval_flag:
            try:
                img_show = result2mask(imgpath, seg_result)
                # out_path = os.path.join(mask_file, os.path.basename(imgpath))
                out_path = os.path.join(cfg.output, pathname.split('/')[-2], 'mask', os.path.basename(imgpath).split('.')[0]+'.png')
                cv2.imwrite(out_path, img_show)
            except:
                print('failed...')
                pass
            
            if out_flag:
                img_show = result2image(imgpath, seg_result)
                out_path = os.path.join(cfg.output, pathname.split('/')[-2], 'pred', os.path.basename(imgpath).split('.')[0]+'.png')
                cv2.imwrite(out_path, img_show)

        if json_flag:
            try:
                result = result2json(img_id, seg_result)
                results.append(result)
            except:
                pass

        if output_source != None:
            if mask_flag:
                img_show = result2mask(imgpath, seg_result)
            else:
                img_show = result2image(imgpath, seg_result)
            out_path = os.path.join(output_source, os.path.basename(imgpath))
            cv2.imwrite(out_path, img_show)

    if json_flag:
        re_js = json.dumps(results)
        fjson = open("eval_masks.json","w")
        fjson.write(re_js)
        fjson.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=cfg.model_path)
    parser.add_argument('--input', type=str, default=cfg.input_source)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--gt_source', type=str, default=cfg.gt_source)
    parser.add_argument('--mask_output', type=str, default=cfg.output)
    parser.add_argument('--resolution', type=int, nargs=2, default=[1920, 1080])
    parser.add_argument('--interact-segm', action="store_true")
    parser.add_argument('--speed-test', action="store_true")
    parser.add_argument('--out', action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--json', action="store_true")
    parser.add_argument('--mask', action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    print(args)

    if args.interact_segm:
        interact_segm(args.model, args.input)
        sys.exit()

    if args.speed_test:
        fps = speed_test(model_path = args.model, resolution=args.resolution)
        sys.exit()

    inference(model_path = args.model,
            input_source = args.input,
            output_source = args.output,
            out_flag = args.out,
            eval_flag = args.eval,
            json_flag = args.json,
            mask_flag=args.mask)
    print('success...')

    if args.eval:
        run_eval_miou(cfg.output)

    if args.json:
        run_eval_mask(
            os.path.join(cfg.dataset.valid_prefix, cfg.dataset.valid_info),
            'eval_masks.json')