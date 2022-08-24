#!/usr/bin/env python

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.lib.io import file_io

import dataset
import models.model
import parameters
import utilities
import visualize

MODES = {
    'show-model-summary': visualize.show_model_summary,
    'plot-model': visualize.plot_model,
    'plot-layer-activations': visualize.plot_layer_activations,
    'train-valid-cm': visualize.train_valid_cm,
}

def infer(params):
    if not (mode := MODES.get(params.mode)):
        raise ValueError(f'unknown mode: {params.mode}')
    
    mode(params)

def get_args():
    import argparse
    ap = argparse.ArgumentParser(description='train')
    
    #######################
    # DATA
    ap.add_argument('--image-files', type=str, required=False, nargs='+', action='append', help='image file to evaluate with model')
    ap.add_argument('--image-size', type=int, required=False, help='single H and W dimension all files are subject to')
    ap.add_argument('--image-limit', type=int, required=False, help='limit number of images to use')
    ap.add_argument('--model-dir', type=str, required=False, help='directory of checkpointed model to load')
    ap.add_argument('--data-dir', type=str, default='./data', help='data directory for reference data')
    ap.add_argument('--save-dir', type=str, default='./data', help='directory for saving data to')
    ap.add_argument('--no-batch', action='store_true', help=r'use single batch for training')
    ap.add_argument('--use-gpu', action=argparse.BooleanOptionalAction, default=True, help=r'use GPU')
    
    #######################
    # MODE
    ap.add_argument('--mode', type=str, choices=['show-model-summary', 'plot-model', 'plot-layer-activations'], help='mode to run')
    
    #######################
    # MODEL
    ap.add_argument('--model', type=str, choices=['cnn'], required=True)
    ap.add_argument('--model-version', type=str, choices=['cnn_v1', 'vgg16_v1', 'vgg16_mpncov_v1'], default='', required=False)
    
    #######################
    # HELP
    ap.add_argument('--describe', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--debug', action=argparse.BooleanOptionalAction, help='enables dumping debug info')
    return ap.parse_args()

def init():
    args = get_args()
    params = parameters.InferParameters(**vars(args))
            
    if not params.use_gpu:
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'

    tf.random.set_seed(42)
    np.random.seed(42)
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
    
    if params.debug:
        tf.debugging.disable_traceback_filtering()
        tf.debugging.set_log_device_placement(True)
    
    return params

def main():
    params = init()
    infer(params)
        
if __name__ == '__main__':
    main()