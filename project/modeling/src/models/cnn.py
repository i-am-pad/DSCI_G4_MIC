#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers, models

def get_model(params):
    '''straight out of the tutorial: https://www.tensorflow.org/tutorials/images/cnn#create_the_convolutional_base
    '''
    model = models.Sequential()
    if params.normalize:
        # normalize pixel's color value to [0, 1] range from [0, 255]
        model.add(layers.Rescaling(1./255, input_shape=(params.image_size, params.image_size, 1)))
    model.add(layers.Conv2D(16, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(32, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(32, 3, activation='relu', padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    #model.add(layers.Dense(1, activation='sigmoid'))
    
    return model