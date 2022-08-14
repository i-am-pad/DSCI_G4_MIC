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
import visualize

def train(params, model, data_split):
    checkpoints_path = os.path.join(params.save_dir,
                                    datetime.now().strftime('%Y%m%d_%H%M%S'),
                                    utilities.get_model_train_param_detail(params, is_checkpoint=True))
    model_detail = utilities.get_model_train_param_detail(params)
    tb_log_path = os.path.join(params.save_dir,
                               f"logs/fit/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{model_detail}")
    
    total_files = sum(data_split['train'].label_counts.values())
    num_classes = float(len(data_split['train'].label_counts))
    class_weights = {
        dataset.G4MicDataGenerator.LABELS[label]:
            1./num_class_instances * total_files/num_classes
        for label, num_class_instances in data_split['train'].label_counts.items()
    }
    logging.info(f'class instances: {data_split["train"].label_counts}')
    logging.info(f'class weights: {class_weights}')
    labels = data_split['train'].LABELS.keys()
    
    history = model.fit(data_split['train'],
                        validation_data = data_split['validation'],
                        epochs = params.epochs,
                        class_weight = class_weights,
                        shuffle = True,
                        callbacks = [
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath = checkpoints_path,
                                monitor='val_accuracy' if not params.multilabel else 'val_categorical_accuracy',
                                mode='max',
                                save_best_only=True,
                            ),
                            # https://www.tensorflow.org/tensorboard/graphs
                            tf.keras.callbacks.TensorBoard(log_dir=tb_log_path),
                            tf.keras.callbacks.EarlyStopping(
                                monitor="val_loss",
                                min_delta=0.001,
                                patience=3,
                                verbose=1 if params.verbose else 0,
                                mode="min",
                                baseline=None,
                                restore_best_weights=False,
                            ),
                            visualize.MultiLabelConfusionMatrixPrintCallback(labels),
                            visualize.MultiLabelConfusionMatrixPlotCallback(params, labels),
                        ],
                        max_queue_size = params.max_queue_size,
                        workers = params.workers,
                        # n.b. this seems to cause a lot of sadness when engaged, beware
                        use_multiprocessing = params.use_multiprocessing,
                        verbose = params.verbose,
                        )
    
    
    if len(data_split['test']):
        split = 'test'
        results = model.evaluate(data_split['test'], verbose=2 if params.verbose else 0, 
                                 return_dict=True,
                                 max_queue_size = params.max_queue_size,
                                 workers = params.workers,
                                 use_multiprocessing = params.use_multiprocessing,
                                 )
    else:
        split = 'validation'
        results = model.evaluate(data_split['validation'], verbose=2 if params.verbose else 0, 
                                 return_dict=True,
                                 max_queue_size = params.max_queue_size,
                                 workers = params.workers,
                                 use_multiprocessing = params.use_multiprocessing,
                                 )
    
    logging.info(f'{split} results: {results}')
    
    if params.multilabel:
        labels = data_split[split].LABELS.keys()
        visualize.print_multilabel_confusion_matrix_singular(split, results['multilabel_cm'], labels)
        save_path = os.path.join(params.save_dir, f'{datetime.now().strftime("%Y%m%d_%H%M%S")}_{split}_confusion_matrix_{model_detail}.png')
        visualize.plot_multilabel_confusion_matrix(results['multilabel_cm'], labels, save_chart=True, save_path=save_path)
        logging.info(f'saved {split} confusion matrix plot to {save_path}')

def get_args():
    import argparse
    ap = argparse.ArgumentParser(description='train')
    
    #######################
    # DATA
    ap.add_argument('--data-dir', type=str, default='./data', help=r'data directory for reference data')
    ap.add_argument('--save-dir', type=str, default='./data', help=r'directory for saving data to')
    ap.add_argument('--image-limit', type=int, default=0, help=r'limit number of images to use')
    ap.add_argument('--image-size', type=int, default=648, help=r'input image dimension for H and W')
    ap.add_argument('--crop-size', type=int, default=0, help='number of bytes to crop across the H axis starting from the head. cropped bytes count is rounded up to the nearest multiple of --image-size. *PE headers generally will not generally exceed 1024 bytes, so this is limited to that value but still with --image-size rounding.')
    ap.add_argument('--no-batch', action='store_true', help=r'use single batch for training')
    ap.add_argument('--use-gpu', action=argparse.BooleanOptionalAction, default=True, help=r'use GPU')
    ap.add_argument('--workers', type=int, default=1, help=r'number of workers for data loading')
    ap.add_argument('--use-multiprocessing', action=argparse.BooleanOptionalAction, default=False, help=r'use multiprocessing for data loading')
    ap.add_argument('--max-queue-size', type=int, default=10, help=r'maximum number of queued samples')
    
    #######################
    # MODEL
    ap.add_argument('--model', type=str, choices=['cnn', 'lr'], required=True)
    ap.add_argument('--model-version', type=str, choices=['cnn_v1', 'vgg16_v1', 'vgg16_mpncov_v1', 'vgg16_mpncov_multilabel_v1', 'lr_v1', 'svc_v1'], default='', required=False)
    
    # training
    ap.add_argument('--optimizer', type=str, default='adam', help='model optimization algorithm selected from https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile')
    ap.add_argument('--learning-rate', type=float, default=0.001, help='learning rate for optimizer')
    ap.add_argument('--weight-decay', type=float, default=0.0, help='weight decay for optimizer')
    ap.add_argument('--trial', type=str, default='trial', help='qualifier between experiments used in saved artifacts if --save-model-evaluation is enabled')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--dropout', type=float, default=0.2, help='dropout rate for base classifier regularization')
    ap.add_argument('--multilabel', action=argparse.BooleanOptionalAction, default=False, help='use multi-label classification')
    
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
    
    data_split = dataset.load_generators(params)
    
    # just need one dataset split for access to summary dataset information
    model = models.model.get_model(params, data_split['train'])
    
    if params.verbose or params.describe:
        utilities.summary_plus(model)
        if params.describe:
            return
    
    train(params, model, data_split)

if __name__ == '__main__':
    main()