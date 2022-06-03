#!/usr/bin/env python

from io import BytesIO
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io
import tensorflow_addons as tfa

def to_gpu(v):
    raise NotImplementedError('TODO: tensorflow version of this')

def read_file(path, dtype='uint8'):
    # TODO: this yields a file much larger in size than it should. why?
    #fin = BytesIO(file_io.read_file_to_string(path, binary_mode=True))
    #return np.frombuffer(fin.read(), dtype=dtype)
    return np.load(path).astype(dtype)

def write_file(path, data):
    np.save(path, data)

def load_model(model_dir, has_f1=True):
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