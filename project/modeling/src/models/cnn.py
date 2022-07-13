#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers, metrics
import tensorflow_addons as tfa

from . import MPNCOV

def get_model_v1(params):
    model = CNN(params)
    channels = 1
    model.build(input_shape=(params.batch_size, params.image_size, params.image_size, channels))
    model.compile(optimizer=params.optimizer,
                  loss='categorical_crossentropy',
                  metrics=[
                      'accuracy',
                      metrics.Precision(),
                      metrics.Recall(),
                      tfa.metrics.F1Score(num_classes=2),
                  ],
                  run_eagerly=params.debug,
                  )
    return model

def get_model_vgg16_mpncov_v1(params):
    model = VGG16_MPNCOV(params)
    channels = 3
    model.build(input_shape=(params.batch_size, params.image_size, params.image_size, channels))
    model.compile(optimizer=params.optimizer,
                  loss='categorical_crossentropy',
                  metrics=[
                      'accuracy',
                      metrics.Precision(),
                      metrics.Recall(),
                      tfa.metrics.F1Score(num_classes=2),
                  ],
                  run_eagerly=params.debug,
                  )
    return model

MODEL_VERSION = {
    '': get_model_v1,
    'cnn_v1': get_model_v1,
    'vgg16_mpncov_v1': get_model_vgg16_mpncov_v1,
}

class CNN(tf.keras.Model):
    '''https://www.tensorflow.org/tutorials/images/cnn#create_the_convolutional_base
    '''
    def __init__(self, params):
        super(CNN, self).__init__()
        rescale_factor = 1./255 if params.normalize else 1.
        self._rescaling = layers.Rescaling(rescale_factor, input_shape=(params.image_size, params.image_size, 1))
        self._conv1 = layers.Conv2D(16, 3, activation='relu', padding='same')
        self._maxpool1 = layers.MaxPooling2D((2, 2), padding='same')
        self._conv2 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self._maxpool2 = layers.MaxPooling2D((2, 2), padding='same')
        self._conv3 = layers.Conv2D(32, 3, activation='relu', padding='same')
        self._flatten = layers.Flatten()
        self._fc1 = layers.Dense(64, activation='relu')
        self._classifier = layers.Dense(2, activation='softmax')

    def call(self, inputs):
        h = self._rescaling(inputs)
        h = self._conv1(h)
        h = self._maxpool1(h)
        h = self._conv2(h)
        h = self._maxpool2(h)
        h = self._conv3(h)
        h = self._flatten(h)
        h = self._fc1(h)
        h = self._classifier(h)
        return 

class VGG16_MPNCOV(tf.keras.Model):
    '''a deep convolutional network that combines VGG-16 and MPNCOV
    '''
    def __init__(self, params):
        super(VGG16_MPNCOV, self).__init__()
        self._features = tf.keras.applications.VGG16(include_top=False,
                                                     weights='imagenet' if params.use_imagenet_weights else None,
                                                     input_shape=(params.image_size,
                                                                  params.image_size,
                                                                  # vgg16 requires 3 channels
                                                                  3))
        self._mpncov = MPNCOV.MPNCOV(params)
        self._classifier = tf.keras.Sequential(
            layers = [
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.Dense(512, activation='relu'),
                layers.Dense(2, activation='softmax'),
            ],
            name='classifier',
        )

    def call(self, inputs):
        h = self._features(inputs)
        # TODO: why is this producing NaNs?
        h = self._mpncov(h)
        h = self._classifier(h)
        return h