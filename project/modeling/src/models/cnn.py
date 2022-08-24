#!/usr/bin/env python

import math
import tensorflow as tf
from tensorflow.keras import layers, metrics
import tensorflow_addons as tfa

from . import MPNCOV
import sys
sys.path.append('../')
import utilities

def get_model_v1(params, compile=True):
    model = CNN(params)
    channels = 1
    model.build(input_shape=(1 if params.no_batch else params.batch_size, params.image_size, params.image_size, channels))
    if compile:
        model.compile(optimizer=params.optimizer,
                    loss='binary_crossentropy',
                    metrics=[
                        'accuracy',
                        metrics.Precision(),
                        metrics.Recall(),
                        tf.keras.metrics.AUC(),
                        tfa.metrics.F1Score(num_classes=1),
                    ],
                    run_eagerly=params.debug,
                    )
    return model

def get_model_vgg16_v1(params, compile=True):
    model = VGG16(params)
    channels = 3
    model.build(input_shape=(1 if params.no_batch else params.batch_size, params.image_size, params.image_size, channels))
    num_classes = 1
    if compile:
        model.compile(optimizer=params.optimizer,
                    loss='binary_crossentropy',
                    metrics=[
                        'accuracy',
                        metrics.Precision(),
                        metrics.Recall(),
                        #tf.keras.metrics.AUC(multi_label=params.multilabel, num_labels=num_classes),
                        tfa.metrics.F1Score(num_classes=num_classes, threshold=params.threshold),
                        utilities.MultiLabelConfusionMatrix(name='multilabel_cm', num_classes=num_classes, threshold=params.threshold),
                    ],
                    run_eagerly=params.debug,
                    )
    return model

def get_model_vgg16_mpncov_v1(params, dataset=None, compile=True):
    model = VGG16_MPNCOV(params, dataset)
    channels = 3
    model.build(input_shape=(1 if params.no_batch else params.batch_size, params.image_size, params.image_size, channels))
    
    if params.multilabel:
        num_classes = len(dataset.label_counts)
        accuracy_metric = metrics.BinaryAccuracy(name='accuracy', threshold=params.threshold)
    else:
        num_classes = 1
        accuracy_metric = 'accuracy'
    if compile:
        model.compile(optimizer=params.optimizer,
                    loss='binary_crossentropy',
                    metrics=[
                        accuracy_metric,
                        metrics.Precision(thresholds=params.threshold),
                        metrics.Recall(thresholds=params.threshold),
                        #tf.keras.metrics.AUC(multi_label=params.multilabel, num_labels=num_classes, thresholds=params.threshold),
                        tfa.metrics.F1Score(num_classes=num_classes, threshold=params.threshold),
                        utilities.MultiLabelConfusionMatrix(name='multilabel_cm', num_classes=num_classes, threshold=params.threshold),
                    ],
                    run_eagerly=params.debug,
                    )
    return model

MODEL_VERSION = {
    '': lambda p, _, compile: get_model_v1(p, compile),
    'cnn_v1': lambda p, _, compile: get_model_v1(p, compile),
    'vgg16_v1': lambda p, _, compile: get_model_vgg16_v1(p, compile),
    'vgg16_mpncov_v1': lambda p, _, compile: get_model_vgg16_mpncov_v1(p, compile=compile),
    'vgg16_mpncov_multilabel_v1': lambda p, dataset, compile: get_model_vgg16_mpncov_v1(p, dataset, compile),
}

class Classifier(tf.keras.Model):
    def __init__(self, params):
        super(Classifier, self).__init__()
        self._params = params
        self._classifier = tf.keras.Sequential(
            layers = [
                layers.Flatten(),
                layers.Dropout(params.dropout_p),
                layers.Dense(64, activation='relu'),
                layers.Dense(1, activation='sigmoid'),
            ],
            name='classifier',
        )
    
    def call(self, inputs):
        return self._classifier(inputs)

class MultiLabelClassifier(tf.keras.Model):
    def __init__(self, params, dataset):
        super(MultiLabelClassifier, self).__init__()
        self._params = params
        self._classifier = tf.keras.Sequential(
            layers = [
                layers.Flatten(),
                layers.Dropout(params.dropout_p),
                layers.Dense(64, activation='relu'),
                # still sigmoid, because this is multilabel; we want to predict a 
                # probability for # each label. softmax is not appropriate here, 
                # since isn't to predict a single best label given a proba distribution.
                layers.Dense(len(dataset.label_counts), activation='sigmoid'),
            ],
            name='multilabel_classifier',
        )
    
    def call(self, inputs):
        return self._classifier(inputs)

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
        self._classifier = layers.Dense(1, activation='sigmoid')

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

class VGG16(tf.keras.Model):
    '''a deep convolutional network wrapping the keras VGG-16 model
    '''
    def __init__(self, params):
        super(VGG16, self).__init__()
        channels = 3
        self._rescaling = layers.Rescaling(1./255, input_shape=(params.image_size, params.image_size, channels))
        self._vgg16 = tf.keras.applications.VGG16(include_top=False,
                                                  weights='imagenet' if params.use_imagenet_weights else None,
                                                  input_shape=(params.image_size, params.image_size, channels))
        if params.use_imagenet_weights:
            self.freeze()
        self._classifier = Classifier(params)

    def call(self, inputs):
        h = self._rescaling(inputs)
        h = self._vgg16(inputs)
        h = self._classifier(h)
        return h
    
    def freeze(self, layers=0):
        for layer in self._vgg16.layers[:layers]:
            layer.trainable = False
    
    def unfreeze(self, layers=0):
        for layer in self._vgg16.layers[layers:]:
            layer.trainable = True

class VGG16_MPNCOV(tf.keras.Model):
    '''a deep convolutional network that combines VGG-16 and MPNCOV
    '''
    def __init__(self, params, dataset=None):
        super(VGG16_MPNCOV, self).__init__()
        channels = 3
        
        h_lengths_to_crop = math.ceil(params.crop_size / params.image_size) if params.crop_size else 0
        self._cropping = layers.Cropping2D(cropping=((h_lengths_to_crop,0), (0,0)))
        h_dim = params.image_size - h_lengths_to_crop
        
        self._rescaling = layers.Rescaling(1./255, input_shape=(h_dim, params.image_size, channels))
        self._features = tf.keras.applications.VGG16(include_top=False,
                                                     weights='imagenet' if params.use_imagenet_weights else None,
                                                     input_shape=(h_dim, params.image_size, channels))
        if params.use_imagenet_weights:
            self.freeze()
        self._mpncov = MPNCOV.MPNCOV(params)
        
        if params.multilabel:
            self._classifier = MultiLabelClassifier(params, dataset)
        else:
            self._classifier = Classifier(params)

    def call(self, inputs):
        h = self._cropping(inputs)
        h = self._rescaling(h)
        h = self._features(h)
        h = self._mpncov(h)
        h = self._classifier(h)
        return h

    def freeze(self, layers=0):
        for layer in self._features.layers[:layers]:
            layer.trainable = False
    
    def unfreeze(self, layers=0):
        for layer in self._features.layers[layers:]:
            layer.trainable = True