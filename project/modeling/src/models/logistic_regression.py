#!/usr/bin/env python

import tensorflow as tf
from tensorflow.keras import layers, metrics, regularizers
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
    model = SVC(params)
    channels = 1
    model.build(input_shape=(1 if params.no_batch else params.batch_size, params.image_size, params.image_size, channels))
    if compile:
        model.compile(optimizer=params.optimizer,
                    loss='hinge',
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

class SVC(tf.keras.Model):
    '''https://stackoverflow.com/questions/54414392/convert-sklearn-svm-svc-classifier-to-keras-implementation
    '''
    def __init__(self, params):
        super(SVC, self).__init__()
        self._params = params
        self._classifier = tf.keras.Sequential(
            layers = [
                layers.Flatten(),
                layers.Dense(1, kernel_regularizer=regularizers.l2(params.svc_l2)),
                # instead of using linear activation that can produce [-inf,+inf],
                # sigmoid squashes the result to [0, 1]. this is important when using
                # the precision and recall metrics, which assume output values in [0, 1].
                layers.Activation('sigmoid'),
            ],
            name='svc',
        )
    
    def call(self, inputs):
        return self._classifier(inputs)