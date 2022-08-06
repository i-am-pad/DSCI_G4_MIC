#!/usr/bin/env python

from datetime import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
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
                                    f'{params.model_version}_{params.image_size}x{params.image_size}_{params.trial}_{{epoch}}-{params.epochs}e_{params.batch_size}b_{params.learning_rate}lr_{params.weight_decay}wd_{params.use_imagenet_weights}imnet')
    tb_log_path = os.path.join(params.save_dir,
                               f"logs/fit/{datetime.now().strftime('%F%m%d-%H%M%S')}_{params.model_version}_{params.image_size}x{params.image_size}_{params.trial}_{params.epochs}e_{params.batch_size}b_{params.learning_rate}lr_{params.weight_decay}wd_{params.use_imagenet_weights}imnet")
    
    # if using the full dataset as of 2022-07-25:
    # malicious = 204855, benign = 33773, total = 238628
    # 1/benign    * total/2 = 3.5328
    # 1/malicious * total/2 = 0.5824
    class_weights = {
        dataset.G4MicDataGenerator.LABELS[label]:
            (1./num_files) * sum(data_split['train'].label_counts.values())/2.
        for label, num_files in data_split['train'].label_counts.items()
    }
    logging.info(f'class instances: {data_split["train"].label_counts}')
    logging.info(f'class weights: {class_weights}')
    
    history = model.fit(data_split['train'],
                        validation_data = data_split['validation'],
                        epochs = params.epochs,
                        class_weight = class_weights,
                        shuffle = True,
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
                                monitor="val_loss",
                                min_delta=0.001,
                                patience=3,
                                verbose=1 if params.verbose else 0,
                                mode="max",
                                baseline=None,
                                restore_best_weights=False,
                            ),
                        ],
                        max_queue_size = params.max_queue_size,
                        workers = params.workers,
                        # n.b. this seems to cause a lot of sadness when engaged, beware
                        use_multiprocessing = params.use_multiprocessing,
                        verbose = params.verbose,
                        )
    
    if len(data_split['test']):
        test_loss, test_acc, test_p, test_r, test_f1 = model.evaluate(data_split['test'], verbose=2 if params.verbose else 0)
        logging.info(f'loss: {test_loss}, accuracy: {test_acc}, precision: {test_p}, recall: {test_r}, f1: {test_f1}')
    else:
        test_loss, test_acc, test_p, test_r, test_f1 = model.evaluate(data_split['validation'], verbose=2 if params.verbose else 0)
        logging.info(f'loss: {test_loss}, accuracy: {test_acc}, precision: {test_p}, recall: {test_r}, f1: {test_f1}')

def get_args():
    import argparse
    ap = argparse.ArgumentParser(description='train')
    
    #######################
    # DATA
    
    ap.add_argument('--data-dir', type=str, default='./data', help=r'data directory for reference data')
    ap.add_argument('--save-dir', type=str, default='./data', help=r'directory for saving data to')
    ap.add_argument('--image-limit', type=int, default=0, help=r'limit number of images to use')
    ap.add_argument('--image-size', type=int, default=648, help=r'input image dimension for H and W')
    ap.add_argument('--no-batch', action='store_true', help=r'use single batch for training')
    ap.add_argument('--use-gpu', action=argparse.BooleanOptionalAction, default=True, help=r'use GPU')
    ap.add_argument('--workers', type=int, default=1, help=r'number of workers for data loading')
    ap.add_argument('--use-multiprocessing', action=argparse.BooleanOptionalAction, default=False, help=r'use multiprocessing for data loading')
    ap.add_argument('--max-queue-size', type=int, default=10, help=r'maximum number of queued samples')
    
    #######################
    # MODEL
    ap.add_argument('--model', type=str, choices=['cnn', 'lr'], required=True)
    ap.add_argument('--model-version', type=str, choices=['cnn_v1', 'vgg16_v1', 'vgg16_mpncov_v1', 'lr_v1', 'svc_v1'], default='', required=False)
    
    # training
    ap.add_argument('--optimizer', type=str, default='adam', help='model optimization algorithm selected from https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile')
    ap.add_argument('--learning-rate', type=float, default=0.001, help='learning rate for optimizer')
    ap.add_argument('--weight-decay', type=float, default=0.0, help='weight decay for optimizer')
    ap.add_argument('--trial', type=str, default='trial', help='qualifier between experiments used in saved artifacts if --save-model-evaluation is enabled')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate for base classifier regularization')
    
    # cnn
    ap.add_argument('--create-channel-dummies', type=bool, action=argparse.BooleanOptionalAction, help='create dummy channels for each image')
    ap.add_argument('--use-imagenet-weights', type=bool, action=argparse.BooleanOptionalAction, help='use imagenet weights for VGG16 backbone')
    ap.add_argument('--dimension-reduction', type=int, default=None, help='dimension reduction for MPNCONV')
    
    # svc
    ap.add_argument('--svc-l2', type=float, default=0.01, help='l2 regularization for svc')
    
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
    np.random.seed(42)
        
    if not params.use_gpu:
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
        for device in visible_devices:
            assert device.device_type != 'GPU'

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