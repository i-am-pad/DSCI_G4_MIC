#!/usr/bin/env python

from io import BytesIO
import numpy as np
from tensorflow.python.lib.io import file_io

def to_gpu(v):
    raise NotImplementedError('TODO: tensorflow version of this')

def read_file(path, dtype='uint8'):
    # TODO: this yields a file much larger in size than it should. why?
    #fin = BytesIO(file_io.read_file_to_string(path, binary_mode=True))
    #return np.frombuffer(fin.read(), dtype=dtype)
    return np.load(path).astype(dtype)

def write_file(path, data):
    np.save(path, data)