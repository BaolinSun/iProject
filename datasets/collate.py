'''
Author: sunbaolin
Date: 2022-02-10 17:26:46
LastEditors: sunbaolin
LastEditTime: 2022-02-19 21:13:11
FilePath: /iProject/datasets/collate.py
Description: 

'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence
from .container import DataContainer

from pprint import pprint


def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, Sequence):
        raise TypeError(f'{batch.dtype} is not supported.')

    if isinstance(batch[0], DataContainer):
        assert len(batch) % samples_per_gpu == 0
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)

                if batch[i].pad_dims is not None:
                    ndim = batch[i].dim()
                    assert ndim > batch[i].pad_dims
                    max_shape = [0 for _ in range(batch[i].pad_dims)]
                    for dim in range(1, batch[i].pad_dims + 1):
                        max_shape[dim - 1] = batch[i].size(-dim)
                    for sample in batch[i:i + samples_per_gpu]:
                        for dim in range(0, ndim - batch[i].pad_dims):
                            assert batch[i].size(dim) == sample.size(dim)
                        for dim in range(1, batch[i].pad_dims + 1):
                            max_shape[dim - 1] = max(max_shape[dim - 1],
                                                     sample.size(-dim))
                    padded_samples = []
                    for sample in batch[i:i + samples_per_gpu]:
                        pad = [0 for _ in range(batch[i].pad_dims * 2)]
                        for dim in range(1, batch[i].pad_dims + 1):
                            pad[2 * dim -
                                1] = max_shape[dim - 1] - sample.size(-dim)
                        padded_samples.append(
                            F.pad(
                                sample.data, pad, value=sample.padding_value))
                    stacked.append(default_collate(padded_samples))
                elif batch[i].pad_dims is None:
                    stacked.append(
                        default_collate([
                            sample.data
                            for sample in batch[i:i + samples_per_gpu]
                        ]))
                else:
                    raise ValueError(
                        'pad_dims should be either None or integers (1-3)')

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], Mapping):
        transposed = zip(*batch)
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)


        
    

    # pprint(batch)
    # print('=============================================================================================================================================================================================')
    # for i in range(len(batch)):
    #     pprint(batch[i])
    #     print('############################################################################################################################################')
    # print('---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------')




"""
[{'gt_bboxes': DataContainer(tensor([[436.9012,  78.5986, 558.7672, 389.5532],
        [  0.0000, 295.4305,  68.7513, 335.6403]])),
  'gt_labels': DataContainer(tensor([1, 1])),
  'gt_masks': DataContainer(tensor([[[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]],

        [[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8)),
  'img': DataContainer(tensor([[[ 0.0056, -0.0629, -0.2684,  ...,  0.0000,  0.0000,  0.0000],
         [-0.1314, -0.0801, -0.3712,  ...,  0.0000,  0.0000,  0.0000],
         [-0.1486, -0.1143, -0.4397,  ...,  0.0000,  0.0000,  0.0000],
         ...,
         [ 0.2624,  0.3138,  0.2796,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.2796,  0.3309,  0.2967,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.3138,  0.3309,  0.3138,  ...,  0.0000,  0.0000,  0.0000]],

        [[ 0.2052,  0.1702, -0.0224,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.2402,  0.3102, -0.0049,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.1877,  0.2752, -0.0574,  ...,  0.0000,  0.0000,  0.0000],
         ...,
         [-0.2150, -0.1625, -0.1800,  ...,  0.0000,  0.0000,  0.0000],
         [-0.1975, -0.1450, -0.1450,  ...,  0.0000,  0.0000,  0.0000],
         [-0.1450, -0.1099, -0.1275,  ...,  0.0000,  0.0000,  0.0000]],

        [[-0.4624, -0.6541, -0.7761,  ...,  0.0000,  0.0000,  0.0000],
         [-0.6018, -0.5670, -0.7936,  ...,  0.0000,  0.0000,  0.0000],
         [-0.6193, -0.5844, -0.8807,  ...,  0.0000,  0.0000,  0.0000],
         ...,
         [-0.5147, -0.4624, -0.4624,  ...,  0.0000,  0.0000,  0.0000],
         [-0.5321, -0.4973, -0.4798,  ...,  0.0000,  0.0000,  0.0000],
         [-0.5670, -0.5147, -0.5147,  ...,  0.0000,  0.0000,  0.0000]]])),
  'img_metas': DataContainer({'filename': './data/coco/val2017/000000397133.jpg', 'ori_shape': (427, 640, 3), 'img_shape': (480, 719, 3), 'pad_shape': (480, 736, 3), 'scale_factor': 1.1241217798594847, 'flip': False, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}})},
 {'gt_bboxes': DataContainer(tensor([[365.9215, 195.7682, 444.6953, 415.8617],
        [ 10.9794, 187.3570, 146.6131, 440.1982],
        [572.4561, 192.0785, 710.0187, 432.9308]])),
  'gt_labels': DataContainer(tensor([1, 1, 1])),
  'gt_masks': DataContainer(tensor([[[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]],

        [[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]],

        [[0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         ...,
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0],
         [0, 0, 0,  ..., 0, 0, 0]]], dtype=torch.uint8)),
  'img': DataContainer(tensor([[[ 1.0844,  0.6392, -0.1657,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.0844,  0.6221, -0.1657,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.0844,  0.6221, -0.1486,  ...,  0.0000,  0.0000,  0.0000],
         ...,
         [ 0.8104,  0.8447,  0.1939,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.2899,  1.2899,  0.4337,  ...,  0.0000,  0.0000,  0.0000],
         [ 0.8276,  0.8789,  0.7591,  ...,  0.0000,  0.0000,  0.0000]],

        [[ 1.4657,  1.0280,  0.1527,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.4657,  1.0105,  0.1527,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.4657,  1.0105,  0.1527,  ...,  0.0000,  0.0000,  0.0000],
         ...,
         [ 1.0980,  1.0980,  0.4153,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.6057,  1.5707,  0.6604,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.1155,  1.1331,  0.9755,  ...,  0.0000,  0.0000,  0.0000]],

        [[ 1.9777,  1.4548,  0.8099,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.9777,  1.4374,  0.8099,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.9777,  1.4374,  0.8099,  ...,  0.0000,  0.0000,  0.0000],
         ...,
         [ 1.5942,  1.5594,  0.8099,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.9603,  1.8905,  0.9319,  ...,  0.0000,  0.0000,  0.0000],
         [ 1.3851,  1.4025,  1.2282,  ...,  0.0000,  0.0000,  0.0000]]])),
  'img_metas': DataContainer({'filename': './data/coco/val2017/000000252219.jpg', 'ori_shape': (428, 640, 3), 'img_shape': (480, 718, 3), 'pad_shape': (480, 736, 3), 'scale_factor': 1.1214953271028036, 'flip': False, 'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True}})}]

"""
