#!/usr/bin/env python

'''used to create local image dataset for the project from a combination of the following:
    1. benign and malicious samples from PDF dataset
    2. malicious samples from Sorel-20M
        * and only malicious; they don't provide any benign samples
    3. benign samples from a Debian 10, Windows 11, and set of conda toolchain files
'''

from io import BytesIO
from functools import reduce
import glob
import numpy as np
from operator import mul
import os
import pandas as pd
import tensorflow as tf
from tensorflow.python.lib.io import file_io
from tqdm import tqdm
from typing import Dict, List, Tuple

def read_file(path: str,
              dtype: str = 'uint8',
              ) -> np.array:
    '''reads a file from GCS into a numpy array
    
    args
        path: GCS path to file in gs://bucket/prefix/filename format
        dtype: dtype to shape data to
    
    returns
        numpy array containing file's content represented as :param:`dtype` bytes
    '''
    fin = BytesIO(file_io.read_file_to_string(path, binary_mode=True))
    return np.frombuffer(fin.read(), dtype='uint8')

def write_file(data: np.array,
               path: str,
               ext: str = '.npy',
               ):
    '''writes a numpy array of the specified dtype to a file in a bucket on GCS
    
    args
        data: numpy array containing bytes to write to GCS
        path: path on GCS to save the file to
        ext: extension to use for saved files. default is numpy's canonical .npy.
    '''
    pre, _ = os.path.splitext(path)
    np.save(file_io.FileIO(pre + ext, 'w'), data)

def convert_to_image(data: np.array,
                     dimension: int,
                     ):
    '''implements file conversion from binary to square bytes used for grayscale image representation
    
    args:
        data: a numpy array containing a file's contents with dtype=uint8
        dimension: value used for H and W
    
    returns
        a numpy 2D numpy array containing the truncated or zero-padded bytes of shape (dimension, dimension)
    '''
    target_shape = (dimension, dimension, 1)
    total_bytes_allowed = reduce(mul, target_shape)
    image = np.zeros(shape=(total_bytes_allowed,))
    num_bytes = min(data.shape[0], total_bytes_allowed)
    image[:num_bytes] = data[:num_bytes]
    
    return image

def convert_to_images(src_dirs: List[str],
                      dst_dir: str,
                      dimension: int,
                      do_overwrite: bool = False,
                      raise_exceptions: bool = True,
                      ) -> List[Tuple[str, str]]:
    '''
    '''
    failures = []
    for src_dir in src_dirs:
        #files = os.listdir(src_dir)
        files = glob.glob(os.path.join(src_dir, '*'))
        for file in tqdm(files, desc=f'converting {src_dir}', unit='files'):
            try:
                dst_path = os.path.join(dst_dir, file)
                
                if not do_overwrite and tf.io.gfile.exists(dst_path):
                    continue
                
                src_path = os.path.join(src_dir, file)    
                data = read_file(src_path)
                image_data = convert_to_image(data, dimension)
                
                write_file(image_data, dst_path)
            except Exception as ex:
                if raise_exceptions:
                    raise
                failures.append((src_path, str(ex)))
    
    return failures

if __name__ == '__main__':
    base_src_dir = '/mnt/z/g4_mic/raw'
    base_dst_dir = '/mnt/z/g4_mic/preprocessed/images'
    
    benign_dirs = [
        os.path.join(base_src_dir, 'benign_samples/benign'),
        os.path.join(base_src_dir, 'pdfmalware/benign'),
    ]
    convert_to_images(benign_dirs, os.path.join(base_dst_dir, 'benign'), dimension=256, do_overwrite=True)
    
    malicious_dirs = [
        os.path.join(base_src_dir, 'pdfmalware/malicious'),
        os.path.join(base_src_dir, 'sorel20m/malicious'),
    ]
    convert_to_images(malicious_dirs, os.path.join(base_dst_dir, 'malicious'), dimension=256, do_overwrite=True)
    