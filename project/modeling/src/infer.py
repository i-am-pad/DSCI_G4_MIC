#!/usr/bin/env python

import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow_addons as tfa

import dataset
from models import cnn
import parameters
import utilities

def plot_layer_activations(params, model, data_split, images_per_row=16):
    '''extremely helpful reference; one (main) chunk is verbatim!
    
    https://towardsdatascience.com/visualizing-intermediate-activations-of-a-cnn-trained-on-the-mnist-dataset-2c34426416c8
    '''
    
    # all the conv / pooling layers only, which means this is super specific to cnn.get_model, and will need
    # to change to something more generic when we use this for other models next term
    layers = model.layers[1:6]
    am = tf.keras.Model(inputs=model.input, outputs=[l.output for l in layers])
    am_layer_names = [l.name for l in layers]
    
    image, _ = list(data_split)[0]

    save_path = os.path.join(params.save_dir,
                            f'{os.path.basename(params.model_dir)}_original_image.png')
    plt.imsave(save_path, image.numpy().reshape(params.image_size, params.image_size), vmin=0, vmax=255, cmap='gray')
    
    activations = am.predict(image)
    
    for layer_ix, (layer_name, activation) in enumerate(zip(am_layer_names, activations)):
        num_features = activation.shape[-1]
        size = activation.shape[1]
        
        num_cols = num_features // images_per_row
        display_grid = np.zeros((size * num_cols, size * images_per_row), dtype='float16')
        
        for c_ix in range(num_cols):
            for r_ix in range(images_per_row):
                #print(c_ix, r_ix, c_ix * images_per_row + r_ix, activation.shape)
                channel_image = activation[0, :, :, c_ix * images_per_row + r_ix]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[c_ix*size:(c_ix + 1)*size,
                                r_ix*size:(r_ix + 1)*size
                                ] = channel_image
        
        scale = 1. / size
        
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        save_path = os.path.join(params.save_dir,
                                    f'{os.path.basename(params.model_dir)}_{layer_ix}_{layer_name}_activations.png')
        plt.imsave(save_path, display_grid, cmap='viridis')

def infer(params, model, data_split):
    if params.plot_layer_activations:
        plot_layer_activations(params, model, data_split)

def get_args():
    import argparse
    ap = argparse.ArgumentParser(description='train')
    
    #######################
    # DATA
    ap.add_argument('--image-files', type=str, required=True, nargs='+', action='append', help='image file to evaluate with model')
    ap.add_argument('--image-size', type=int, required=True, help='single H and W dimension all files are subject to')
    ap.add_argument('--model-dir', type=str, required=True, help='directory of checkpointed model to load')
    ap.add_argument('--save-dir', type=str, default='./data', help='directory for saving data to')
    ap.add_argument('--use-gpu', type=bool, action=argparse.BooleanOptionalAction, default=False)
    
    #######################
    # MODE
    ap.add_argument('--plot-layer-activations', type=bool, action=argparse.BooleanOptionalAction, help='saves a plot of the layer activations ')
    
    #######################
    # HELP
    ap.add_argument('--describe', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--debug', action=argparse.BooleanOptionalAction, help='enables dumping debug info')
    return ap.parse_args()

def main():
    tf.random.set_seed(42)
    
    args = get_args()
    params = parameters.InferParameters(**vars(args))
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
    
    if params.debug:
        tf.debugging.disable_traceback_filtering()
    
    model = utilities.load_model(params.model_dir)
    
    if params.verbose or params.describe:
        # just uses print, and logging.info produces some ugly stuff if used with print_fn arg
        model.summary()
        if params.describe:
            return
    
    data = dataset.load_from_files(params)
    
    infer(params, model, data)
        
if __name__ == '__main__':
    main()