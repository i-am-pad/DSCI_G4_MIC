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
    checkpoints_path = os.path.join(params.save_dir,
                                    f'{params.model}_{params.image_size}x{params.image_size}_{params.trial}_{{epoch}}-{params.epochs}e_{params.batch_size}b')
    
    history = model.fit(data_split['train'],
                        validation_data = data_split['validation'],
                        epochs = params.epochs,
                        
                        # TODO: this doesn't work! how does this argument actually work?
                        # class weight for benign, which there are 2x fewer of
                        #class_weight = {0: params.class_weight},
                        
                        # TODO: data generator needs to implement on_epoch_end
                        #       to use this
                        #shuffle = True,
                        
                        callbacks = [
                            tf.keras.callbacks.ModelCheckpoint(
                                filepath = checkpoints_path,
                                monitor='val_accuracy',
                                mode='max',
                                save_best_only=True,
                            )
                        ],
                        
                        verbose = params.verbose,
                        )
    
    test_loss, test_acc, test_p, test_r, test_f1 = model.evaluate(data_split['test'], verbose=2 if params.verbose else 0)
    logging.info(f'loss: {test_loss}, accuracy: {test_acc}, precision: {test_p}, recall: {test_r}, f1: {test_f1}')
    
    if params.save_model_evaluation:
        save_plots(params, history)

def save_plots(params, history):
    logging.info('saving training metrics plots...')
    # TODO: this is way too generic to produce anything intelligible. it needs to be
    #       multiple axes, separated by split (train, validation, test)
    for metric in history.history.keys():
        plt.plot(history.history[metric], label=metric)
    
    plt.ylim([0.5, 1])
    plt.title('Training Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(params.save_dir, f'{params.model}_{params.trial}_{params.epochs}e_{params.batch_size}b.png'))

def get_args():
    import argparse
    ap = argparse.ArgumentParser(description='train')
    
    #######################
    # DATA
    
    ap.add_argument('--data-dir', type=str, default='./data', help=r'data directory for reference data')
    ap.add_argument('--save-dir', type=str, default='./data', help=r'directory for saving data to')
    ap.add_argument('--image-limit', type=float, default=0, help=r'proportional limit between 0 (1%) and 1 (100%) for number of images to load. default value 0 loads all images.')
    ap.add_argument('--image-size', type=int, default=648, help=r'input image dimension for H and W')
    ap.add_argument('--use-gpu', type=bool, action=argparse.BooleanOptionalAction, default=False)
    
    #######################
    # MODEL
    
    ap.add_argument('--model', type=str, choices=['cnn'], required=True)
    ap.add_argument('--optimizer', type=str, default='adam', help='default is adam. see the following for alternatives: https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile')
    
    # training
    ap.add_argument('--trial', type=str, default='trial', help='qualifier between experiments used in saved artifacts if --save-model-evaluation is enabled')
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--batch-size', type=int, default=1)
    ap.add_argument('--class-weight', type=int, default=2, help='imbalance factor applied to benign class, which there are 2x fewer of')
    ap.add_argument('--save-model-evaluation', type=bool, action=argparse.BooleanOptionalAction, default=True, help='saves model evaluation information to --save-dir following training')
    
    # cnn
    # ...
    
    #######################
    # HELP
    ap.add_argument('--describe', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--verbose', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    ap.add_argument('--debug', action=argparse.BooleanOptionalAction, help='enables dumping debug info')
    return ap.parse_args()

def main():
    tf.random.set_seed(42)
    
    args = get_args()
    params = parameters.TrainParameters(**vars(args))
    
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
    
    if params.debug:
        tf.debugging.disable_traceback_filtering()
    
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
                  metrics=[
                      'accuracy',
                      metrics.Precision(),
                      metrics.Recall(),
                      tfa.metrics.F1Score(num_classes=2)
                      ],
                  run_eagerly=params.debug,
                  )
    
    # TODO: torch -> tf params.use_gpu? do you just get this for free if detected w/ tensorflow?
    #if params.use_gpu:
    #    model = utilities.to_gpu(model)
    
    # this fails due to OOM on host (GPU) pretty easily!
    #data = dataset.load(params)
    #data_split = dataset.train_valid_test_split(params, data)
    
    # this loads only single batches of images in memory at a time,
    # but much less likely to run into OOM issues.
    data_split = dataset.load_generators(params)
    
    train(params, model, data_split)

if __name__ == '__main__':
    main()