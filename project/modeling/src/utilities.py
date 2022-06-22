#!/usr/bin/env python

from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import tensorflow_addons as tfa

from io import BytesIO
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.python.lib.io import file_io

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

def load_model(model_dir, has_f1=True) -> tf.keras.Model:
    '''loads a model from a tensorflow format save directory
    
    args
        model_dir: path to the directory of a model checkpoint to load
        has_f1: enables loading external library's F1 score metric
    
    returns
        target model trained to the point of when the checkpoint was saved
    '''
    custom_objects = None
    if has_f1:
        def f1(y_true, y_pred):
            '''wrapper to help loading keras models saved with tensorflow_addons F1Score metric
            '''
            metric = tfa.metrics.F1Score(num_classes=3, threshold=0.5)
            metric.update_state(y_true, y_pred)
            return metric.result()
        custom_objects = {'f1': f1}
    
    return tf.keras.models.load_model(model_dir.rstrip('/'), compile=False, custom_objects=custom_objects)