#!/usr/bin/env python

import logging
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow_addons as tfa

import dataset
from models import cnn
import parameters
import utilities

def train(params, model, data_split):
    history = model.fit(data_split['train'],
                        validation_data = data_split['validation'],
                        epochs = params.epochs,
                        # class weight for benign, which there are 2x fewer of
                        class_weight = {0: params.class_weight},
                        shuffle = True,
                        verbose = params.verbose,
                        )
    
    if params.show_model_evaluation:
        plt.plot(history.history['accuracy'], label='accuracy')
        plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.5, 1])
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(params.save_dir, f'{params.model}_{params.trial}_{params.epochs}e_{params.batch_size}b.png'))

        test_loss, test_acc, test_p, test_r, test_f1 = model.evaluate(data_split['test'], verbose=2 if params.verbose else 0)
        
    # TODO
    #if params.save_model_train_data:
    #    ...

def get_args():
    import argparse
    ap = argparse.ArgumentParser(description='train')
    
    #######################
    # DATA
    
    ap.add_argument('--data-dir', type=str, default='./data', help=r'data directory for reference data')
    ap.add_argument('--save-dir', type=str, default='./data', help=r'directory for saving data to')
    ap.add_argument('--image-limit', type=int, default=0, help=r'limit for number of images to load. default value 0 loads all images.')
    ap.add_argument('--image-size', type=int, default=648, help=r'input image dimension for H and W')
    ap.add_argument('--use-gpu', type=bool, action=argparse.BooleanOptionalAction, default=False)
    
    #######################
    # MODEL
    
    ap.add_argument('--model', type=str, choices=['cnn'], required=True)
    ap.add_argument('--mode', type=str, choices=['train', 'eval'])
    ap.add_argument('--optimizer', type=str, default='adam', help='default is adam. see the following for alternatives: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile')
    
    # training
    ap.add_argument('--trial', type=str, default='trial', help='qualifier between experiments used in saved artifacts if --save-model-train-data is enabled')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--class-weight', type=int, default=2, help='imbalance factor applied to benign class, which there are 2x fewer of')
    ap.add_argument('--show-model-evaluation', type=bool, action=argparse.BooleanOptionalAction, default=True, help='produces model evaluation information following training')
    
    # cnn
    # ...
    
    #######################
    # HELP
    ap.add_argument('--describe', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    return ap.parse_args()

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
    tf.random.set_seed(42)
    
    args = get_args()
    params = parameters.Parameters(**vars(args))
    
    if params.model == 'cnn':
        model = cnn.get_model(params)
    
    model.build((None, params.image_size, params.image_size, 1))
    
    if params.verbose or params.describe:
        # just uses print, and logging.info produces some ugly stuff if used with print_fn arg
        model.summary()
        if params.describe:
            return
    
    model.compile(optimizer=params.optimizer,
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy', metrics.Precision(), metrics.Recall(), tfa.metrics.F1Score(num_classes=2)],
                  )
    
    if params.use_gpu:
        model = utilities.to_gpu(model)
    
    data = dataset.load(params)
    data_split = dataset.train_valid_test_split(params, data)
    
    if params.mode == 'train':
        train(params, model, data_split)
    elif params.mode == 'eval':
        pass
    elif params.mode == 'test':
        return
    else:
        return

if __name__ == '__main__':
    main()