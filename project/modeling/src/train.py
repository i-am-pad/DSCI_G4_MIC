#!/usr/bin/env python

import logging
import torch
from tqdm import tqdm, trange
import tensorflow as tf

import dataset
from models import cnn
import parameters
import utilities

def train(model, params, train_dataloader): # TODO: , eval_dataloader):
    model.train()
    
    # TODO: use other optimizer?
    optimizer = tf.keras.optimizers.Adam(model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay)
    
    trainer, reporter = model.trainer_reporter(optimizer)
    # len(torch.utils.data.DataLoader) for an iterable dataset like 
    # dataset.SEPTACustomerComplaints is in batch units
    num_batches = len(train_dataloader)
    
    for _ in trange(params.epochs, unit='epoch', desc='epochs'):
        for batch_ix, batch in enumerate(tqdm(train_dataloader, unit='batch', desc='batches', total=num_batches)):
            results = trainer(batch_ix, batch)
            
            if params.progress_report_cadence and batch_ix % params.progress_report_cadence == 0:
                tqdm.write(reporter(*results, kind=f'train progress batch {batch_ix}/{num_batches}'))
    
    logging.info(reporter(*results, kind='train final'))
    
    # TODO
    #if params.save_model_train_data:
    #    ...

def get_args():
    import argparse
    ap = argparse.ArgumentParser(description='train')
    
    #######################
    # DATA
    ap.add_argument('--data-dir', type=str, default='./data', help=r'data directory for reference data and output')
    ap.add_argument('--image-limit', type=int, default=0, help=r'limit for number of images to load. default value 0 loads all images.')
    ap.add_argument('--image-size', type=int, default=648, help=r'input image dimension for H and W')
    ap.add_argument('--use-gpu', type=bool, action=argparse.BooleanOptionalAction, default=False)
    
    #######################
    # MODEL
    
    ap.add_argument('--model', type=str, choices=['cnn'], required=True)
    ap.add_argument('--mode', type=str, choices=['train', 'eval'])
    
    # training
    ap.add_argument('--trial', type=str, default='trial', help='qualifier between experiments used in saved artifacts if --save-model-train-data is enabled')
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--save-model-train-data', type=bool, action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument('--progress-report-cadence', type=int, default=1_000, help='period in batches for logging training status information. 0 disables it.')
    
    # cnn
    # ...
    
    #######################
    # HELP
    ap.add_argument('--describe', action=argparse.BooleanOptionalAction, help='prints model architecture and exits')
    return ap.parse_args()

def main():
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
    torch.manual_seed(42)
    
    args = get_args()
    params = parameters.Parameters(**vars(args))
    
    data = dataset.SEPTACustomerComplaints(params)
    
    if params.model == 'cnn':
        model = cnn.CNN(params)
    
    if params.use_gpu:
        model = utilities.to_gpu(model)
    
    logging.info(model)
    if args.describe:
        return
    
    dataloaders = dataset.train_valid_test_split(params, data)
    
    if params.mode == 'train':
        train(model, params, dataloaders['train'])
    elif params.mode == 'eval':
        pass
    else:
        return

if __name__ == '__main__':
    main()