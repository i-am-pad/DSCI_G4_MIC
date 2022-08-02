#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers, metrics
import tensorflow_addons as tfa

def get_model_lr_v1(params, compile=True):
    model = LogisticRegression(params)
    channels = 1
    model.build(input_shape=(1 if params.no_batch else params.batch_size, params.image_size, params.image_size, channels))
    if compile:
        model.compile(optimizer=params.optimizer,
                    loss='binary_crossentropy',
                    metrics=[
                        'accuracy',
                        metrics.Precision(),
                        metrics.Recall(),
                        tf.keras.metrics.AUC(from_logits=True),
                        tfa.metrics.F1Score(num_classes=1),
                    ],
                    run_eagerly=params.debug,
                    )
    return model

def get_model_svc_v1(params, compile=True):
    model = LogisticRegression(params)
    channels = 1
    model.build(input_shape=(1 if params.no_batch else params.batch_size, params.image_size, params.image_size, channels))
    if compile:
        model.compile(optimizer=params.optimizer,
                    loss='squared_hinge',
                    metrics=[
                        'accuracy',
                        metrics.Precision(),
                        metrics.Recall(),
                        tf.keras.metrics.AUC(from_logits=True),
                        tfa.metrics.F1Score(num_classes=1),
                    ],
                    run_eagerly=params.debug,
                    )
    return model

MODEL_VERSION = {
    '': lambda p, compile: get_model_lr_v1(p, compile),
    'lr_v1': lambda p, compile: get_model_lr_v1(p, compile),
    'svc_v1': lambda p, compile: get_model_svc_v1(p, compile),
}

class LogisticRegression(tf.keras.Model):
    def __init__(self, params):
        super(LogisticRegression, self).__init__()
        self._params = params
        self._classifier = tf.keras.Sequential(
            layers = [
                layers.Flatten(),
                layers.Dense(1, activation='sigmoid'),
            ],
            name='logistic_regression',
        )
    
    def call(self, inputs):
        return self._classifier(inputs)