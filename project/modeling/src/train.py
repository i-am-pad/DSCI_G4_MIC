#!/usr/bin/env python

from datetime import datetime
import logging
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras import metrics
import tensorflow_addons as tfa

import dataset
import models.model
import parameters
import utilities

def train(params, model, data_split):
    checkpoints_path = os.path.join(params.save_dir,
                                    f'{params.model}_{params.image_size}x{params.image_size}_{params.trial}_{{epoch}}-{params.epochs}e_{params.batch_size}b_{params.learning_rate}lr_{params.weight_decay}wd_{params.use_imagenet_weights}imnet')
    tb_log_path = os.path.join(params.save_dir,
                               f"logs/fit/{datetime.now().strftime('%F%m%d-%H%M%S')}")
    
    history = model.fit(data_split['train'],
                        validation_data = data_split['validation'],
                        epochs = params.epochs,
                        class_weight = {0: params.class_weight, 1: 1.},
                        shuffle = True,
                        
                        # TODO: data generator needs to implement on_epoch_end
                        #       to use this
                        #shuffle = True,
                        
                        callbacks = [
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath = checkpoints_path,
                                monitor='val_accuracy',
                                mode='max',
                                save_best_only=True,
                            ),
                            
                            # https://www.tensorflow.org/tensorboard/graphs
                            tf.keras.callbacks.TensorBoard(log_dir=tb_log_path, ),
                            
                            tf.keras.callbacks.EarlyStopping(
                                monitor="loss",
                                min_delta=0.001,
                                patience=1,
                                verbose=1 if params.verbose else 0,
                                mode="auto",
                                baseline=None,
                                restore_best_weights=False,
                            ),
                        ],
                        
                        verbose = params.verbose,
                        )
    
    test_loss, test_acc, test_p, test_r, test_f1 = model.evaluate(data_split['test'], verbose=2 if params.verbose else 0)
    logging.info(f'loss: {test_loss}, accuracy: {test_acc}, precision: {test_p}, recall: {test_r}, f1: {test_f1}')

def get_args():
    import argparse
    ap = argparse.ArgumentParser(description='train')
    
    #######################
    # DATA
    
    ap.add_argument('--data-dir', type=str, default='./data', help=r'data directory for reference data')
    ap.add_argument('--save-dir', type=str, default='./data', help=r'directory for saving data to')
    ap.add_argument('--image-limit', type=float, default=0, help=r'proportional limit between 0 and 1 for number of images to load. default value 0 loads all images.')
    ap.add_argument('--image-size', type=int, default=648, help=r'input image dimension for H and W')
    
    #######################
    # MODEL
    
    ap.add_argument('--model', type=str, choices=['cnn'], required=True)
    ap.add_argument('--model-version', type=str, choices=['cnn_v1', 'vgg16_v1', 'vgg16_mpncov_v1'], default='', required=False)
    
    # training
    ap.add_argument('--optimizer', type=str, default='adam', help='model optimization algorithm selected from https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile')
    ap.add_argument('--learning-rate', type=float, default=0.001, help='learning rate for optimizer')
    ap.add_argument('--weight-decay', type=float, default=0.0, help='weight decay for optimizer')
    ap.add_argument('--trial', type=str, default='trial', help='qualifier between experiments used in saved artifacts if --save-model-evaluation is enabled')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--class-weight', type=float, default=2., help='imbalance factor applied to benign class')
    
    # cnn
    ap.add_argument('--create-channel-dummies', type=bool, action=argparse.BooleanOptionalAction, help='create dummy channels for each image')
    ap.add_argument('--use-imagenet-weights', type=bool, action=argparse.BooleanOptionalAction, help='use imagenet weights for VGG16 backbone')
    ap.add_argument('--dimension-reduction', type=int, default=None, help='dimension reduction for MPNCONV')
    
    #######################
    # HELP
    ap.add_argument('--describe', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--debug', action=argparse.BooleanOptionalAction, help='enables dumping debug info')
    return ap.parse_args()

def init():
    args = get_args()
    params = parameters.TrainParameters(**vars(args))
    
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
    
    model = models.model.get_model(params)
    
    if params.verbose or params.describe:
        utilities.summary_plus(model)
        if params.describe:
            return
    
    data_split = dataset.load_generators(params)
    
    train(params, model, data_split)

if __name__ == '__main__':
    main()