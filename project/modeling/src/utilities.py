#!/usr/bin/env python

from io import BytesIO
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.lib.io import file_io

import parameters

def read_file(path: str, dtype: str = 'uint8') -> np.array:
    '''reads a file from GCS into a numpy array
    
    args
        path: GCS path to file in gs://bucket/prefix/filename format
        dtype: dtype to shape data to
    
    returns
        numpy array containing file's content represented as :param:`dtype` bytes
    '''
    fin = BytesIO(file_io.read_file_to_string(path, binary_mode=True))
    return np.load(fin).astype(dtype)

def write_file(data: np.array, path: str, ext: str = '.npy') -> None:
    '''writes a numpy array of the specified dtype to a file in a bucket on GCS
    
    args
        data: numpy array containing bytes to write to GCS
        path: path on GCS to save the file to
        ext: extension to use for saved files. default is numpy's canonical .npy.
    '''
    pre, _ = os.path.splitext(path)
    np.save(file_io.FileIO(pre + ext, 'w'), data)

def summary_plus(layer, i=0):
    '''shows summary of base model's nested layers
    
    https://stackoverflow.com/questions/58748048/how-to-print-all-activation-shapes-more-detailed-than-summary-for-tensorflow/58752908#58752908
    '''
    if hasattr(layer, 'layers'):
        if i != 0: 
            layer.summary()
        for l in layer.layers:
            i += 1
            summary_plus(l, i=i)

def get_model_train_param_detail(params: parameters.TrainParameters):
    '''returns string detailing model training parameterization
    
    args
        params: model training parameters
    
    returns
        string representation of model training parameterization
    
    raises
        TypeError if :param:`params` is not a valid model training parameters object
    '''
    if type(params) != parameters.TrainingParameters:
        raise TypeError(f'params must be of type parameters.TrainingParameters, not {type(params)}')
    
    return f'{params.model_version}_{params.image_size}x{params.image_size}_{params.trial}_{{epoch}}-{params.epochs}e_{params.batch_size}b_{params.learning_rate}lr_{params.weight_decay}wd_{params.use_imagenet_weights}imnet'