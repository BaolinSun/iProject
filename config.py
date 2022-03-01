'''
Author: sunbaolin
Contact: baolin.sun@mail.sdu.edu.cn
Date: 2022-02-07 18:40:55
LastEditors: sunbaolin
LastEditTime: 2022-02-28 20:34:36
Description: file content
FilePath: /iProject/config.py
'''

from utils import Config

COCO_LABEL = [1]

COCO_CLASSES = ('person', )

COCO_LABEL_MAP = {1: 1}

CLASS_NAMES = (COCO_CLASSES, COCO_LABEL)

dataset_base = Config({
    'name': 'Base Dataset',

    # Training images and annotations
    'train_images': './data/coco/images/',
    'train_info': 'path_to_annotation_file',

    # Validation images and annotations.
    'valid_images': './data/coco/images/',
    'valid_info': 'path_to_annotation_file',

    # Whether or not to load GT. If this is False, eval.py quantitative evaluation won't work.
    'has_gt': True,

    # A list of names for each of you classes.
    'class_names': COCO_CLASSES,

    # COCO class ids aren't sequential, so this is a bandage fix. If your ids aren't sequential,
    # provide a map from category_id -> index in class_names + 1 (the +1 is there because it's 1-indexed).
    # If not specified, this just assumes category ids start at 1 and increase sequentially.
    'label_map': None
})

backbone_base = Config({
    'name': 'Base Backbone',
    'path': 'path/to/checkpoints/weights',
    'type': None,
})

resnet18_backbone = backbone_base.copy({
    'name': 'resnet18',
    'path': './checkpoints/resnet18-5c106cde.pth',
    'type': 'ResNetBackbone',
    'num_stages': 4,
    'frozen_stages': 1,
    'out_indices': (0, 1, 2, 3)
})

resnet34_backbone = backbone_base.copy({
    'name': 'resnet34',
    'path': './checkpoints/resnet34-333f7ec4.pth',
    'type': 'ResNetBackbone',
    'num_stages': 4,
    'frozen_stages': 1,
    'out_indices': (0, 1, 2, 3)
})

resnet50_backbone = backbone_base.copy({
    'name': 'resnet50',
    'path': './checkpoints/resnet50-19c8e357.pth',
    'type': 'ResNetBackbone',
    'num_stages': 4,
    'frozen_stages': 1,
    'out_indices': (0, 1, 2, 3)
})

resnet101_backbone = backbone_base.copy({
    'name': 'resnet101',
    'path': './checkpoints/resnet101-5d3b4d8f.pth',
    'type': 'ResNetBackbone',
    'num_stages': 4,
    'frozen_stages': 1,
    'out_indices': (0, 1, 2, 3)
})

resnet152_backbone = backbone_base.copy({
    'name': 'resnet152',
    'path': './checkpoints/resnet152-b121ed2d.pth',
    'type': 'ResNetBackbone',
    'num_stages': 4,
    'frozen_stages': 1,
    'out_indices': (0, 1, 2, 3)
})

fpn_base = Config({
    'name': 'fpn',
    # 'in_channels': [64, 128, 256, 512],
    'in_channels': [256, 512, 1024, 2048],
    'out_channels': 256,
    'start_level': 0,
    'num_outs': 5,
})

coco2017_dataset = dataset_base.copy({
    'name': 'COCO2017',

    'train_prefix': './data/coco/',
    'train_info': 'annotations/instances_train2022.json',
    'trainimg_prefix': 'train2017/',
    'train_images': './data/coco/',

    'valid_prefix': './data/coco/',
    'valid_info': 'annotations/instances_val2022.json',
    'validimg_prefix': 'val2017/',
    'valid_images': './data/coco/',
    
    'label_map': COCO_LABEL_MAP
})

coco_base_config = Config({
    'dataset': coco2017_dataset,
    'num_classes': 2,  # This should include the background class
})

model_base_config = coco_base_config.copy({
    'name': 'ins_his',

    'imgs_per_gpu': 8,
    'workers_per_gpu': 1,
    'num_gpus': 1,

    # backbone
    'backbone': resnet101_backbone,

    # fpn
    'fpn': fpn_base,

    # Dataset
    'dataset': coco2017_dataset,
    'num_classes': len(coco2017_dataset.class_names) + 1,

    # pretrained
    'pretrained': 'checkpoints/runtime/model_resnet101_epoch_72.pth',
    'epoch_iters_start': 73,
    'epoch_iters_total': 144,

    # learn rate
    'lr_config': dict(base_lr=0.01, step=[27, 33, 52], warmup='linear', warmup_iters=500, warmup_ratio=0.01),

    'train_show_interval': 5,

    'test_cfg': dict(
                nms_pre=500,
                score_thr=0.1,
                mask_thr=0.5,
                update_thr=0.05,
                kernel='gaussian',  # gaussian/linear
                sigma=2.0,
                max_per_img=30),

    'model_path': 'checkpoints/runtime/model_resnet50_epoch_8.pth',
    'input_source': 'data/coco/val2022',
    'output_source': None
})

cfg = model_base_config.copy()