'''
Author: sunbaolin
Date: 2022-02-10 14:24:10
LastEditors: sunbaolin
LastEditTime: 2022-02-16 21:21:55
FilePath: /iProject/datasets/piplines.py
Description: 

'''

import os
import cv2
import numbers
import torch
import numpy as np
import pycocotools.mask as maskUtils

from compose import Compose
from collections.abc import Sequence
from .container import DataContainer as DC

# ============================== CLASS ================================================


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
        results['bbox_fields'].append('gt_bboxes')

        return results

    def _load_masks(self, results):
        h, w = results['img_info']['height'], results['img_info']['width']
        gt_masks = results['ann_info']['masks']
        if self.poly2mask:
            gt_mask = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_mask
        results['mask_fields'].append('gt_masks')
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)
            if results is None:
                return None
        if self.with_label:
            results = self._load_labels(results)
        if self.with_mask:
            results = self._load_masks(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += ('(with_bbox={}, with_label={}, with_mask={})').format(
            self.with_bbox, self.with_label, self.with_mask)
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
        for key in results.get('bbox_fields', []):
            bboxes = results[key] * results['scale_factor']
            bboxes[:, 0::2] = np.clip(bboxes[:, 0::2], 0, img_shape[1] - 1)
            bboxes[:, 1::2] = np.clip(bboxes[:, 1::2], 0, img_shape[0] - 1)
            results[key] = bboxes

    def _resize_masks(self, results):
        for key in results.get('mask_fields', []):
            if results[key] is None:
                continue
            if self.keep_ratio:
                masks = [
                    imrescale(mask,
                            results['scale_factor'],
                            interpolation='nearest') for mask in results[key]
                ]
            else:
                raise NotImplementedError

            results[key] = np.stack(masks)

    def __call__(self, results):
        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_masks(results)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += '(img_scale={}, multiscale_mode={}, ratio_range={}, keep_ratio={})'.format(
            self.img_scale, self.multiscale_mode, self.ratio_range,
            self.keep_ratio)
        return repr_str


class RandomFlip(object):
    """Flip the image & bbox & mask."""

    def __init__(self, flip_ratio=None, direction='horizontal') -> None:
        self.flip_ratio = flip_ratio
        self.direction = direction
        if flip_ratio is not None:
            assert flip_ratio >= 0 and flip_ratio <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        if 'flip' not in results:
            flip = True if np.random.rand() < self.flip_ratio else False
            results['flip'] = flip
        if 'flip_direction' not in results:
            results['flip_direction'] = self.direction
        if results['flip']:
            # flip image
            results['img'] = self.img_flip(results['img'], direction=results['flip_direction'])
            # flip bboxes
            for key in results.get('bbox_fields', []):
                results[key] = self.bbox_flip(results[key], results['img_shape'], results['flip_direction'])
            # flip masks
            for key in results.get('mask_fields', []):
                results[key] = np.stack([self.img_flip(mask, direction=results['flip_direction']) for mask in results[key]])
            # flip segs
            for key in results.get('seg_fields', []):
                results[key] = self.img_flip(results[key], direction=results['flip_direction'])

        return results


    def img_flip(self, img, direction='horizontal'):
        """Flip an image horizontally or vertically."""

        assert direction in ['horizontal', 'vertical', 'diagonal']
        if direction == 'horizontal':
            return np.flip(img, axis=1)
        elif direction == 'vertical':
            return np.flip(img, axis=0)
        else:
            return np.flip(img, axis=(0, 1))

    def bbox_flip(self, bboxes, img_shape, direction):
        """Flip bboxes horizontally or vertically."""
        assert bboxes.shape[-1] % 4 == 0
        flipped = bboxes.copy()
        if direction == 'horizontal':
            w = img_shape[1]
            flipped[..., 0::4] = w - bboxes[..., 2::4] - 1
            flipped[..., 2::4] = w - bboxes[..., 0::4] - 1
        elif direction == 'vertical':
            h = img_shape[0]
            flipped[..., 1::4] = h - bboxes[..., 3::4] - 1
            flipped[..., 3::4] = h - bboxes[..., 1::4] - 1
        else:
            raise ValueError(
                'Invalid flipping direction "{}"'.format(direction))
        return flipped


    def __repr__(self):
        return self.__class__.__name__ + '(flip_ratio={})'.format(
            self.flip_ratio)


class Normalize(object):
    """Normalize the image."""

    def __init__(self, mean, std, to_rgb=True) -> None:
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        for key in results.get('img_fields', ['img']):
            results[key] = self.imnormalize(results[key], self.mean, self.std, self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def imnormalize(slef, img, mean, std, to_rgb=True):
        img = img.copy().astype(np.float32)
        assert img.dtype != np.uint8
        mean = np.float64(mean.reshape(1, -1))
        stdinv = 1 / np.float64(std.reshape(1, -1))
        if to_rgb:
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        cv2.subtract(img, mean, img)
        cv2.multiply(img, stdinv, img)
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb={self.to_rgb})'
        return repr_str


class Pad(object):
    """Pad the image & mask."""

    def __init__(self, size=None, size_divisor=None, pad_val=0) -> None:
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        if self.size is not None:
            padded_img = self.img_pad(results['img'], shape=self.size)
        elif self.size_divisor is not None:
            padded_img = self.impad_to_multiple(results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_masks(self, results):
        pad_shape = results['pad_shape'][:2]
        for key in results.get('mask_fields', []):
            
            padded_masks = [
                self.img_pad(mask, shape = pad_shape, pad_val=self.pad_val)
                for mask in results[key]
            ]
            if padded_masks:
                results[key] = np.stack(padded_masks, axis=0)
            else:
                results[key] = np.empty((0, ) + pad_shape, dtype=np.uint8)

    def _pad_seg(self, results):
        for key in results.get('seg_fields', []):
            results[key] = self.img_pad(results[key], results['pad_shape'][:2])


    def __call__(self, results):
        self._pad_img(results)
        self._pad_masks(results)
        self._pad_seg(results)

        return results

    def img_pad(self, img, *, shape=None, padding=None, pad_val=0, padding_mode='constant'):

        assert (shape is not None) ^ (padding is not None)
        if shape is not None:
            padding = (0, 0, shape[1] - img.shape[1], shape[0] - img.shape[0])

        # check pad_val
        if isinstance(pad_val, tuple):
            assert len(pad_val) == img.shape[-1]
        elif not isinstance(pad_val, numbers.Number):
            raise TypeError('pad_val must be a int or a tuple. '
                            f'But received {type(pad_val)}')

        # check padding
        if isinstance(padding, tuple) and len(padding) in [2, 4]:
            if len(padding) == 2:
                padding = (padding[0], padding[1], padding[0], padding[1])
        elif isinstance(padding, numbers.Number):
            padding = (padding, padding, padding, padding)
        else:
            raise ValueError('Padding must be a int or a 2, or 4 element tuple.'
                            f'But received {padding}')

        # check padding mode
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        border_type = {
            'constant': cv2.BORDER_CONSTANT,
            'edge': cv2.BORDER_REPLICATE,
            'reflect': cv2.BORDER_REFLECT_101,
            'symmetric': cv2.BORDER_REFLECT
        }
        img = cv2.copyMakeBorder(
            img,
            padding[1],
            padding[3],
            padding[0],
            padding[2],
            border_type[padding_mode],
            value=pad_val)

        return img

    def impad_to_multiple(self, img, divisor, pad_val=0):
        """Pad an image to ensure each edge to be multiple to some number."""
        pad_h = int(np.ceil(img.shape[0] / divisor)) * divisor
        pad_w = int(np.ceil(img.shape[1] / divisor)) * divisor
        return self.img_pad(img, shape=(pad_h, pad_w), pad_val=pad_val)

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(size={}, size_divisor={}, pad_val={})'.format(
            self.size, self.size_divisor, self.pad_val)
        return repr_str


class DefaultFormatBundle(object):

    def __call__(self, results):

        if 'img' in results:
            img = results['img']
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            img = np.ascontiguousarray(img.transpose(2, 0, 1))
            results['img'] = DC(self.to_tensor(img), stack=True)

        for key in ['proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels']:
            if key not in results:
                continue
            results[key] = DC(self.to_tensor(results[key]))
        
        if 'gt_masks' in results:
            results['gt_masks'] = DC(results['gt_masks'], cpu_only=True)

        return results


    def to_tensor(self, data):
        if isinstance(data, torch.Tensor):
            return data
        elif isinstance(data, np.ndarray):
            return torch.from_numpy(data)
        elif isinstance(data, Sequence) and not isinstance(data, str):
            return torch.tensor(data)
        elif isinstance(data, int):
            return torch.LongTensor([data])
        elif isinstance(data, float):
            return torch.FloatTensor([data])
        else:
            raise TypeError(f'type {type(data)} cannot be converted to tensor.')


    def __repr__(self):
        return self.__class__.__name__


def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.

    Args:
        data (torch.Tensor | numpy.ndarray | Sequence | int | float): Data to
            be converted.
    """

    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not isinstance(data, str):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(f'type {type(data)} cannot be converted to tensor.')

class ImageToTensor(object):
    """Convert image to :obj:`torch.Tensor` by given keys.

    The dimension order of input image is (H, W, C). The pipeline will convert
    it to (C, H, W). If only 2 dimension (H, W) is given, the output would be
    (1, H, W).

    Args:
        keys (Sequence[str]): Key of images to be converted to Tensor.
    """

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        """Call function to convert image in results to :obj:`torch.Tensor` and
        transpose the channel order.

        Args:
            results (dict): Result dict contains the image data to convert.

        Returns:
            dict: The result dict contains the image converted
                to :obj:`torch.Tensor` and transposed to (C, H, W) order.
        """
        for key in self.keys:
            img = results[key]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            results[key] = to_tensor(img.transpose(2, 0, 1))
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(keys={self.keys})'

class Collect(object):

    def __init__(self, keys, meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'img_norm_cfg')) -> None:
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        
        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]
        data['img_metas'] = DC(img_meta, cpu_only=True)

        for key in self.keys:
            data[key] = results[key]

        return data


    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


class TestCollect(object):
    def __init__(self,
                 keys,
                 meta_keys=('filename', 'ori_shape', 'img_shape', 'pad_shape',
                            'scale_factor', 'flip', 'img_norm_cfg')):
                            
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):

        data = {}
        img_meta = {}
        for key in self.meta_keys:
            img_meta[key] = results[key]        
        data['img_metas'] = img_meta
        for key in self.keys:
            data[key] = results[key]        
        return data

    def __repr__(self):
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'


class MultiScaleFlipAug(object):

    def __init__(self, transforms, img_scale, flip=False):
        self.transforms = Compose(transforms)
        self.img_scale = img_scale if isinstance(img_scale,
                                                 list) else [img_scale]
        assert isinstance(self.img_scale, list)
        self.flip = flip

    def __call__(self, results):
        aug_data = []
        flip_aug = [False, True] if self.flip else [False]
        for scale in self.img_scale:
            for flip in flip_aug:
                _results = results.copy()
                _results['scale'] = scale
                _results['flip'] = flip
                data = self.transforms(_results)
                aug_data.append(data)
        # list of dict to dict of list
        aug_data_dict = {key: [] for key in aug_data[0]}
        for data in aug_data:
            for key, val in data.items():
                aug_data_dict[key].append(val)
        return aug_data_dict

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += '(transforms={}, img_scale={}, flip={})'.format(
            self.transforms, self.img_scale, self.flip)
        return repr_str


class LoadImage(object):
    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None 
        img = cv2.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results
        



# ============================== FUNCTION ================================================

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


def imrescale(img, scale, return_scale=False, interpolation='bilinear'):
    h, w = img.shape[:2]
    new_size, scale_factor = rescale_size((w, h), scale, return_scale=True)
    rescaled_img = imresize(img, new_size, interpolation=interpolation)
    if return_scale:
        return rescaled_img, scale_factor
    else:
        return rescaled_img