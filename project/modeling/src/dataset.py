#!/usr/bin/env python

import glob
from itertools import chain
import logging
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import pandas as pd
import tensorflow as tf
from tqdm import tqdm

import parameters
import utilities

class G4MicDataGenerator(tf.keras.utils.Sequence):
    '''this is a work around for the memory limitations of loading the entirety of this dataset.
    it loads images into memory (host or GPU depending on params) only in batch sized blocks.
    '''
    LABELS = {'benign': 0, 'malicious': 1}
    
    def __init__(self, params, split):
        self._params = params
        self._split = split
        self._init_filepaths()

    def _init_filepaths(self):
        self._filepaths = self._init_binary() if not self._params.multilabel else self._init_multilabel()
        
        if type(self._params) in [parameters.TrainParameters, parameters.DatasetParameters]:
            train_size = int(self._params.train_size * len(self._filepaths))
            validation_size = int(self._params.validation_size * len(self._filepaths))
            #test_size = len(self._filepaths) - train_size - validation_size
            
            # TODO: randomize selection of filepaths
            if self._split == 'train':
                self._filepaths = self._filepaths[:train_size]
            elif self._split == 'validation':
                self._filepaths = self._filepaths[train_size : train_size + validation_size]
            elif self._split == 'test':
                self._filepaths = self._filepaths[train_size + validation_size :]
            
            if not self._params.no_batch:
                num_files = len(self._filepaths)
                excess_files = num_files % self._params.batch_size
                self._filepaths = self._filepaths[:num_files - excess_files]
            
            if len(self._filepaths) == 0 and self._split != 'test':
                raise ValueError(f'No files found for split {self._split} with batch size alignment to {self._params.batch_size}. num_files before alignment: {num_files}')
            
            if self._params.verbose:
                logging.info(f'dataset {self._split}: {len(self._filepaths)} files')
        elif type(self._params) == parameters.InferParameters:
            pass
        else:
            raise ValueError(f'Unknown parameters type: {type(self._params)}')

        self.on_epoch_end()

    def _init_multilabel(self):
        '''initializes the baseline benign and malicious classes unless configured
        for multilabel classification. a registry is used to determine the labels
        when in multilabel mode.
        '''
        if not self._params.multilabel:
            return
        
        registry_path = os.path.join(self._params.data_dir, 'registry.csv')
        self._df_r = pd.read_csv(registry_path)
        
        # this is different than the binary case! image limit is applied to the full
        # dataset rather than per class.
        if self._params.image_limit:
            self._df_r = self._df_r.sample(n=self._params.image_limit)
        else:
            # shuffles everything
            self._df_r = self._df_r.sample(frac=1.).reset_index(drop=True)
        
        self._label_counts = {
            l: c
            for l, c in dict(self._df_r.loc[:, [c for c in self._df_r.columns if not c in {'path', 'file', 'base_file', 'kind'}]].sum()).items()
            # TODO: why does this filter break training? it's nice to have when a class isn't represented
            #       due to image_limit. that scenario introduces numeric instability in the metrics.
            #if c
        }
        
        G4MicDataGenerator.LABELS = {
            l: ix
            for ix, l in enumerate(self._label_counts.keys())
        }
        
        return self._df_r.path.values
    
    def _init_binary(self):
        if self._params.multilabel:
            return
        
        G4MicDataGenerator.LABELS = {'benign': 0, 'malicious': 1}
        self._label_counts = {'benign': 0, 'malicious': 0}
        
        filepaths = []
        # TODO: just use the full dataset registry instead
        for label in G4MicDataGenerator.LABELS.keys():
            dir = os.path.join(self._params.data_dir, label)
            registry = os.path.join(self._params.data_dir, f'{label}.txt')
            if tf.io.gfile.exists(os.path.join(registry)):
                with open(registry, 'r') as fin:
                    paths = [os.path.join(dir, fp.strip()) for fp in fin.readlines()]
            else:
                paths = tf.io.gfile.glob(os.path.join(dir, '*'))
            
            filepaths.append(paths)
            self._label_counts[label] = len(paths)
        
        if self._params.image_limit:
            # applies file limit by class
            filepaths = [ fps[:self._params.image_limit] for fps in filepaths ]
            self._label_counts = {label: self._params.image_limit for label in self._label_counts}
        
        filepaths = [ fps for fps in chain.from_iterable(filepaths) ]
        np.random.shuffle(filepaths)
        
        return filepaths

    def __len__(self):
        return len(self._filepaths) if self._params.no_batch else len(self._filepaths) // self._params.batch_size
   
    def __getitem__(self, index):
        indices = self.indices if self._params.no_batch else self.indices[index * self._params.batch_size : (index + 1) * self._params.batch_size]
        images = []
        labels = []
        for ix in indices:
            fp = self._filepaths[ix]
            data = utilities.read_file(fp, dtype='float32')
            channels = 1
            data = tf.convert_to_tensor(data.reshape(self._params.image_size, self._params.image_size, channels), )
            # VGG16 requires images with 3 channels, so this optionally adds dummy channels
            if self._params.create_channel_dummies:
                data = tf.concat([data, data, data], axis=-1)
            images.append(data)
            
            if not self._params.multilabel:
                labels.append(self.LABELS['benign'] if 'benign' in fp else self.LABELS['malicious'])
            else:
                labels.append(self._df_r.iloc[ix][self.LABELS.keys()].values.astype(np.int32))
        
        return (tf.convert_to_tensor(images),
                tf.convert_to_tensor(labels)
               )
    
    @property
    def label_counts(self):
        '''total dataset's label counts before any splitting'''
        return self._label_counts
    
    def on_epoch_end(self):
        self.indices = np.arange(len(self._filepaths))
        np.random.shuffle(self.indices)

def load_generators(params):
    return {
        split: G4MicDataGenerator(params, split)
        for split in ['train', 'validation', 'test']
    }

def load_data(params):
    generators = load_generators(params)
    return {
        split: generators[split].__getitem__(0)
        for split in ['train', 'validation', 'test']
    }

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
        if not tf.io.gfile.exists(to_dir):
            tf.io.gfile.mkdir(to_dir)
        
        files = tf.io.gfile.glob(os.path.join(from_dir, '*'))
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
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
    
    import argparse
    ap = argparse.ArgumentParser(description='''Dataset Resizing Utility
    
    Resizes the dataset --from-image-size to --to-image-size and saves results in a mirror directory with the --to-image-size value included in the top-level directory name as a _HxW suffix.
    
    Examples:
        
            ./dataset.py --data-dir ./data/images --from-image-size 648 --from-image-size 32
            
        This will result in 32x32 results from the 648x648 source in "./data/images" to be produced and saved in a mirrored directory structure under "./data/images_32x32".
        
            ./dataset.py --data-dir ./data/images --save-dir ./data/save/new_images --from-image-size 648 --from-image-size 32
        
        The difference between this and the last command is just --save-dir. This overrides the default behavior for saving data to use the specified directory "./data/save/new_images" instead of creating a directory beside the original with the image dimensions as a suffix.
    ''')
    ap.add_argument('--data-dir', type=str, required=True, help=r'data directory for reference data')
    ap.add_argument('--save-dir', type=str, default=None, help=r'override default behavior by specifying this directory to save model artifacts in')
    ap.add_argument('--image-limit', type=int, default=0, help=r'limit for number of images per class to load. default value 0 loads all images. this is a factor of images by classes.')
    ap.add_argument('--image-size', type=int, default=648, help=r'image dimensions')
    ap.add_argument('--from-image-size', type=int, default=648, help=r'image dimensions')
    ap.add_argument('--to-image-size', type=int, help=r'image dimensions to resize to when --mode=resize')
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

    resize(params)