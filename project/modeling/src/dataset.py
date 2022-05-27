#!/usr/bin/env python

import glob
import logging
import numpy as np
import os
import pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import tensorflow as tf
from tqdm import tqdm

import utilities

def train_valid_test_split(params, dataset: tf.data.Dataset):
    train_size = int(params.train_split * len(dataset))
    validation_size = int(params.valid_split * len(dataset))
    
    result = {}
    result['train'] = dataset.take(train_size).cache().batch(params.batch_size)
    remaining = dataset.skip(train_size)
    result['validation'] = remaining.take(validation_size).cache().batch(params.batch_size)
    result['test'] = remaining.skip(validation_size).cache().batch(params.batch_size)
    
    return result

def load(params):
    label_classes = {'benign': 0, 'malicious': 1}
    
    images = []
    labels = []
    for label_name, label_value in label_classes.items():
        dir = os.path.join(params.data_dir, label_name)
        files = glob.glob(os.path.join(dir, '*'))
        if params.image_limit:
            files = files[:params.image_limit]
        
        labels.extend([label_value] * len(files))
        for file in tqdm(files, desc=f'files ({label_name})', unit='file'):
            data = utilities.read_file(file).astype('float64')
            images.append(data.reshape(params.image_size, params.image_size, 1))
    
    # TODO: should labels be one-hot encoded? these are just label encoded atm.
    dataset = tf.data.Dataset.from_tensor_slices((np.array(images), 
                                                  tf.keras.utils.to_categorical(np.array(labels))))
    
    return dataset

def resize(params):
    label_classes = {'benign': 0, 'malicious': 1}
    
    from_shape = (params.from_image_size, params.from_image_size)
    to_shape = (params.to_image_size, params.to_image_size)
    
    if params.save_dir:
        to_dir_base = params.save_dir
    else:
        to_dir_base = params.data_dir.rstrip('/') + f'_{params.to_image_size}x{params.to_image_size}'
        if not os.path.exists(to_dir_base):
            os.mkdir(to_dir_base)
    
    for label_name, label_value in label_classes.items():
        from_dir = os.path.join(params.data_dir, label_name)
        to_dir = os.path.join(to_dir_base, label_name)
        if not os.path.exists(to_dir):
            os.mkdir(to_dir)
        
        files = glob.glob(os.path.join(from_dir, '*'))
        if params.image_limit:
            files = files[:params.image_limit]
        
        logging.info(f'resizing and copying {len(files)} files from {from_dir} to {to_dir}')
        
        for file in tqdm(files, desc=f'files ({label_name})', unit='file'):
            image = utilities.read_file(file).reshape(*from_shape)
            # np.expand dims is to change from (H, W) shape to (H, W, C) to appease
            # the tf api. C is channel, e.g. 1 for grayscale, 3 for RGB, 4 for RGBA.
            image_resized = tf.image.resize(np.expand_dims(image, -1),
                                            to_shape,
                                            # this is the default
                                            method=tf.image.ResizeMethod.BILINEAR)
            filename = os.path.basename(file)
            to_path = os.path.join(to_dir, filename)
            utilities.write_file(to_path, image_resized)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
    
    import argparse
    ap = argparse.ArgumentParser(description='Dataset')
    ap.add_argument('--data-dir', type=str, default='./data', help=r'data directory for reference data')
    ap.add_argument('--save-dir', type=str, default=None, help=r'override default behavior by specifying this directory to save model artifacts in')
    ap.add_argument('--image-limit', type=int, default=0, help=r'limit for number of images per class to load. default value 0 loads all images. this is a factor of images by classes.')
    ap.add_argument('--image-size', type=int, default=648, help=r'image dimensions')
    ap.add_argument('--from-image-size', type=int, default=648, help=r'image dimensions')
    ap.add_argument('--to-image-size', type=int, help=r'image dimensions to resize to when --mode=resize')
    ap.add_argument('--mode', type=str, choices=['test', 'resize'],
                    help=
    '''sets the mode for the application.
    
    modes:
        test: loads the dataset as specified and prints its metadata. this is the default.
          
        resize: resizes the dataset --from-image-size to --to-image-size and saves results in a mirror directory with the --to-image-size value included in the top-level directory name as a _HxW suffix. for example:
        
            ./dataset.py --data-dir ./data/images --from-image-size 648 --from-image-size 32
            
        this will result in 32x32 results from the 648x648 source in "./data/images" to be produced and saved in a mirrored directory structure under "./data/images_32x32".
        
            ./dataset.py --data-dir ./data/images --save-dir ./data/save/new_images --from-image-size 648 --from-image-size 32
        
        the difference between this and the last command is just --save-dir. this overrides the default behavior for saving data to use the specified directory "./data/save/new_images"
        instead of creating a directory beside the original with the image dimensions as a suffix.
    ''')
    # see https://www.tensorflow.org/api_docs/python/tf/image/ResizeMethod
    # used in dataset.py with https://www.tensorflow.org/api_docs/python/tf/image/resize#args
    # ctrl+f for bilinear in the 2nd link above, and you'll find a section
    # defining all the choices and what they mean. this may be something to
    # look at more closely w.r.t. data loss.
    # TODO: use tf's ResizeMethod enum here, not a string
    #ap.add_argument('--resize-method', type=str, default='nearest', help='tensorflow image.#ResizeMethod value')
    ap.add_argument('--normalize', type=bool, default=True, help=r'squeeze pixel value range to [0, 1]')
    args = ap.parse_args()
    
    from collections import namedtuple
    mock_parameters = namedtuple('Parameters', ' '.join(vars(args).keys()))
    
    params = mock_parameters(**vars(args))
    
    if params.mode == 'test':
        ds = load(params)
        logging.info(f'loaded {len(ds)} files. dataset metadata: {ds}')
    elif params.mode == 'resize':
        resize(params)