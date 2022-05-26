#!/usr/bin/env python

import logging
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

import dataset
import parameters

def get_model(params):
    '''straight out of the tutorial: https://www.tensorflow.org/tutorials/images/cnn#create_the_convolutional_base
    '''
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(params.image_size, params.image_size, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2))
    
    return model