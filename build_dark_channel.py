import os
import torch
from torch.utils.ffi import create_extension

this_file = os.path.dirname(os.path.realpath(__file__))

if torch.cuda.is_available:
    sources = ['src/dark_channel.c']
    headers = ['src/dark_channel.h']
    include_dirs = ['/usr/cuda/include']
    defines = [('WITH_CUDA', None)]
    with_cuda = True

extra_objects = ['src/kernels.o']
extra_objects = [os.path.join(this_file, name) for name in extra_objects]

ffi = create_extension('dark_channel', headers=headers, sources=sources, with_cuda=with_cuda, define_macros=defines, relative_to=__file__, extra_objects=extra_objects, include_dirs=include_dirs)

if __name__ == '__main__':
        ffi.build()
