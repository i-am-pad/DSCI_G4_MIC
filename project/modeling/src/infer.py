#!/usr/bin/env python

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.lib.io import file_io

import dataset
from models import model
import parameters
import utilities
import visualize

def infer(params, model, data_split):
    if params.plot_layer_activations:
        visualize.plot_layer_activations(params, model, data_split)

def get_args():
    import argparse
    ap = argparse.ArgumentParser(description='train')
    
    #######################
    # DATA
    ap.add_argument('--image-files', type=str, required=True, nargs='+', action='append', help='image file to evaluate with model')
    ap.add_argument('--image-size', type=int, required=True, help='single H and W dimension all files are subject to')
    ap.add_argument('--model-dir', type=str, required=True, help='directory of checkpointed model to load')
    ap.add_argument('--save-dir', type=str, default='./data', help='directory for saving data to')
    
    #######################
    # MODE
    ap.add_argument('--plot-layer-activations', type=bool, action=argparse.BooleanOptionalAction, help='saves a plot of the layer activations ')
    
    #######################
    # HELP
    ap.add_argument('--describe', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--debug', action=argparse.BooleanOptionalAction, help='enables dumping debug info')
    return ap.parse_args()

def init():
    args = get_args()
    params = parameters.InferParameters(**vars(args))
    
    tf.random.set_seed(42)
    
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
    
    model = model.load_model(params)
    
    if params.verbose or params.describe:
        # just uses print, and logging.info produces some ugly stuff if used with print_fn arg
        model.summary()
        if params.describe:
            return
    
    data = dataset.load_from_files(params)
    
    infer(params, model, data)
        
if __name__ == '__main__':
    main()