'''
Author: sunbaolin
Contact: baolin.sun@mail.sdu.edu.cn
Date: 2022-02-07 18:43:09
LastEditors: Please set LastEditors
LastEditTime: 2022-02-09 15:02:56
Description: file content
FilePath: /iProject/datasets/utils.py
'''

import os
from unittest import result
import cv2
from matplotlib import scale
from matplotlib.pyplot import isinteractive
import numpy as np

cv2_interp_codes = {
    'nearest': cv2.INTER_NEAREST,
    'bilinear': cv2.INTER_LINEAR,
    'bicubic': cv2.INTER_CUBIC,
    'area': cv2.INTER_AREA,
    'lanczos': cv2.INTER_LANCZOS4
}


def imresize(img,
             size,
             return_scale=False,
             interpolation='bilinear',
             out=None):
    """Resize image to a given size."""
    h, w = img.shape[:2]

    resized_img = cv2.resize(img,
                             size,
                             dst=out,
                             interpolation=cv2_interp_codes[interpolation])

    if not return_scale:
        return resized_img
    else:
        w_scale = size[0] / w
        h_scale = size[1] / h
        return resized_img, w_scale, h_scale


def rescale_size(old_size, scale, return_scale=False):
    """Calculate the new size to be rescaled to."""

    w, h = old_size
    if isinstance(scale, (float, int)):
        if scale <= 0:
            raise ValueError(f'Invalid scale {scale}, must be positive.')
        scale_factor = scale
    elif isinstance(scale, tuple):
        max_long_edge = max(scale)
        max_short_edge = min(scale)
        scale_factor = min(max_long_edge / max(h, w),
                           max_short_edge / min(h, w))
    else:
        raise TypeError(
            f'Scale must be a number or tuple of int, but got {type(scale)}')

    new_size = _scale_size((w, h), scale_factor)

    if return_scale:
        return new_size, scale_factor
    else:
        return new_size


def _scale_size(size, scale):
    """Rescale a size by a ratio."""
    w, h = size
    return int(w * float(scale) + 0.5), int(h * float(scale) + 0.5)


def imrescale(img, scale, return_scale, interpolation='bilinear'):
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img


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


class LoadImageFromFile(object):

    def __init__(self, to_float32=False, color_type='color') -> None:
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        if results['img_prefix'] is not None:
            filename = os.path.join(results['img_prefix'],
                                    results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        img = cv2.imread(filename, cv2.IMREAD_COLOR)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape

        return results

    def __repr__(self) -> str:
        return '{} (to_float32={}, color_type={})'.format(
            self.__class__.__name__, self.to_float32, self.color_type)


class LoadAnnotations(object):

    def __init__(self,
                 with_bbox=True,
                 with_label=True,
                 with_mask=False,
                 poly2mask=True) -> None:
        self.with_bbox = with_bbox
        self.with_label = with_label
        self.with_mask = with_mask
        self.poly2mask = poly2mask

    def _load_labels(self, results):
        results['gt_labels'] = results['ann_info']['labels']
        return results

    def _load_bboxes(self, results):
        anno_info = results['ann_info']
        results['gt_bboxes'] = anno_info['bboxes']

        gt_bboxes_ignore = anno_info.get('bboxes_ignore', None)
        if gt_bboxes_ignore is not None:
            results['gt_bboxes_ignore'] = gt_bboxes_ignore
            results['bbox_fields'].append('gt_bboxes_ignore')
        results['bboxex_fields'].append('gt_bboxed')

        return results
        

    def __call__(self):
        pass

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={})').format(self.with_bbox, self.with_label,self.with_mask)
        return repr_str


class Resize(object):

    def __init__(self,
                 img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True) -> None:
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert isinstance(self.img_scale, list)

        if ratio_range is not None:
            # mode 1: given a scale and a range of image ratio
            assert len(self.img_scale) == 1
        else:
            # mode 2: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        assert isinstance(img_scales, list)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    def _random_scale(self, results):
        if self.ratio_range is not None:
            pass
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            pass
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx

    def _resize_img(self, results):
        if self.keep_ratio:
            img, scale_factor = imrescale(results['img'],
                                          results['scale'],
                                          return_scale=True)
        else:
            raise NotImplementedError

        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio

    def _resize_bboxes(self, results):
        img_shape = results['img_shape']
        # for key in results.get('bbox_fields', [])

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '(img_scale={}, multiscale_mode={}, ratio_range={}, keep_ratio={})'.format(
            self.img_scale, self.multiscale_mode, self.ratio_range,
            self.keep_ratio)
        return repr_str
