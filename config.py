'''
Author: sunbaolin
Contact: baolin.sun@mail.sdu.edu.cn
Date: 2022-02-07 18:40:55
LastEditors: Please set LastEditors
LastEditTime: 2022-02-09 17:22:10
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

coco2017_dataset = dataset_base.copy({
    'name': 'COCO2017',

    'train_prefix': './data/coco/',
    'train_info': 'annotations/instances_val2022.json',
    'trainimg_prefix': 'val2017/',
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
    'name': 'hisense',

    # Dataset
    'dataset': coco2017_dataset,
    'num_classes': len(coco2017_dataset.class_names) + 1,

    'imgs_per_gpu': 2,
    'workers_per_gpu': 1,
    'num_gpus': 1,
})

cfg = model_base_config.copy()