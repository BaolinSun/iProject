'''
Author: sunbaolin
Date: 2022-02-15 09:40:55
LastEditors: sunbaolin
LastEditTime: 2022-02-15 09:46:43
FilePath: /iProject/setup.py
Description: 

'''
import os
import torch

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def make_cuda_ext(name, module, sources):

    define_macros = []

    if torch.cuda.is_available() or os.getenv('FORCE_CUDA', '0') == '1':
        define_macros += [("WITH_CUDA", None)]
    else:
        raise EnvironmentError('CUDA is required to compile MMDetection!')

    return CUDAExtension(
        name='{}.{}'.format(module, name),
        sources=[os.path.join(*module.split('.'), p) for p in sources],
        define_macros=define_macros,
        extra_compile_args={
            'cxx': ['-std=c++14'],
            'nvcc': [
                '-D__CUDA_NO_HALF_OPERATORS__',
                '-D__CUDA_NO_HALF_CONVERSIONS__',
                '-D__CUDA_NO_HALF2_OPERATORS__',
            ]
        })


if __name__ == '__main__':
    setup(name='focalloss',
          version='1.0.0',
          package_data={'models/losses': ['*/*.so']},
          classifiers=[
              'Development Status :: 4 - Beta',
              'License :: OSI Approved :: Apache Software License',
              'Operating System :: OS Independent',
              'Programming Language :: Python :: 3',
              'Programming Language :: Python :: 3.5',
              'Programming Language :: Python :: 3.6',
              'Programming Language :: Python :: 3.7',
          ],
          ext_modules=[
              make_cuda_ext(name='sigmoid_focal_loss_cuda',
                            module='models.losses.sigmoid_focal_loss',
                            sources=[
                                'src/sigmoid_focal_loss.cpp',
                                'src/sigmoid_focal_loss_cuda.cu'
                            ])
          ],
          cmdclass={'build_ext': BuildExtension},
          zip_safe=False)
