#!/usr/bin/env python

from io import BytesIO
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.lib.io import file_io

import parameters

def read_file(path: str, dtype: str = 'uint8') -> np.array:
    '''reads a file from GCS into a numpy array
    
    args
        path: GCS path to file in gs://bucket/prefix/filename format
        dtype: dtype to shape data to
    
    returns
        numpy array containing file's content represented as :param:`dtype` bytes
    '''
    fin = BytesIO(file_io.read_file_to_string(path, binary_mode=True))
    return np.load(fin).astype(dtype)

def write_file(data: np.array, path: str, ext: str = '.npy') -> None:
    '''writes a numpy array of the specified dtype to a file in a bucket on GCS
    
    args
        data: numpy array containing bytes to write to GCS
        path: path on GCS to save the file to
        ext: extension to use for saved files. default is numpy's canonical .npy.
    '''
    pre, _ = os.path.splitext(path)
    np.save(file_io.FileIO(pre + ext, 'w'), data)

def summary_plus(layer, i=0):
    '''shows summary of base model's nested layers
    
    https://stackoverflow.com/questions/58748048/how-to-print-all-activation-shapes-more-detailed-than-summary-for-tensorflow/58752908#58752908
    '''
    if hasattr(layer, 'layers'):
        if i != 0: 
            layer.summary()
        for l in layer.layers:
            i += 1
            summary_plus(l, i=i)

def get_model_train_param_detail(params: parameters.TrainParameters, is_checkpoint=False):
    '''returns string detailing model training parameterization
    
    args
        params: model training parameters
        is_checkpoint: if True, returns string for checkpoint file name that includes epoch
    
    returns
        string representation of model training parameterization
    
    raises
        TypeError if :param:`params` is not a valid model training parameters object
    '''
    if type(params) != parameters.TrainParameters:
        raise TypeError(f'params must be of type parameters.TrainingParameters, not {type(params)}')
    
    if is_checkpoint:
        epoch_string = f'_{{epoch}}-'
    else:
        epoch_string = '_'
    
    return f'{params.model_version}_{params.image_size}x{params.image_size}_{params.crop_size}crop_{params.trial}{epoch_string}{params.epochs}e_{params.batch_size}b_{params.learning_rate}lr_{params.weight_decay}wd_{params.dimension_reduction}dr{"_pretrained"  if params.use_imagenet_weights else ""}_{params.threshold}thr'



'''following is slightly adapted from the following to support using thresholds:
https://github.com/tensorflow/addons/blob/v0.17.0/tensorflow_addons/metrics/multilabel_confusion_matrix.py#L28-L188

original docs:
https://www.tensorflow.org/addons/api_docs/python/tfa/metrics/MultiLabelConfusionMatrix
'''

import warnings
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import Metric
from typeguard import typechecked
from tensorflow_addons.utils.types import AcceptableDTypes, FloatTensorLike

class MultiLabelConfusionMatrix(Metric):
    """Computes Multi-label confusion matrix.
    Class-wise confusion matrix is computed for the
    evaluation of classification.
    If multi-class input is provided, it will be treated
    as multilabel data.
    Consider classification problem with two classes
    (i.e num_classes=2).
    Resultant matrix `M` will be in the shape of `(num_classes, 2, 2)`.
    Every class `i` has a dedicated matrix of shape `(2, 2)` that contains:
    - true negatives for class `i` in `M(0,0)`
    - false positives for class `i` in `M(0,1)`
    - false negatives for class `i` in `M(1,0)`
    - true positives for class `i` in `M(1,1)`
    Args:
        num_classes: `int`, the number of labels the prediction task can have.
        name: (Optional) string name of the metric instance.
        dtype: (Optional) data type of the metric result.
        threshold: (Optional) threshold for the binary prediction values.
    Usage:
    >>> # multilabel confusion matrix
    >>> y_true = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.int32)
    >>> y_pred = np.array([[1, 0, 0], [0, 1, 1]], dtype=np.int32)
    >>> metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=3)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy()  #doctest: -DONT_ACCEPT_BLANKLINE
    array([[[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[0., 1.],
            [1., 0.]]], dtype=float32)
    >>> # if multiclass input is provided
    >>> y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.int32)
    >>> y_pred = np.array([[1, 0, 0], [0, 0, 1]], dtype=np.int32)
    >>> metric = tfa.metrics.MultiLabelConfusionMatrix(num_classes=3)
    >>> metric.update_state(y_true, y_pred)
    >>> result = metric.result()
    >>> result.numpy() #doctest: -DONT_ACCEPT_BLANKLINE
    array([[[1., 0.],
            [0., 1.]],
    <BLANKLINE>
           [[1., 0.],
            [1., 0.]],
    <BLANKLINE>
           [[1., 1.],
            [0., 0.]]], dtype=float32)
    """

    @typechecked
    def __init__(
        self,
        num_classes: FloatTensorLike,
        name: str = "Multilabel_confusion_matrix",
        dtype: AcceptableDTypes = None,
        threshold: float = None,
        **kwargs,
    ):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.threshold = threshold
        self.true_positives = self.add_weight(
            "true_positives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )
        self.false_positives = self.add_weight(
            "false_positives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )
        self.false_negatives = self.add_weight(
            "false_negatives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )
        self.true_negatives = self.add_weight(
            "true_negatives",
            shape=[self.num_classes],
            initializer="zeros",
            dtype=self.dtype,
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        if sample_weight is not None:
            warnings.warn(
                "`sample_weight` is not None. Be aware that MultiLabelConfusionMatrix "
                "does not take `sample_weight` into account when computing the metric "
                "value."
            )

        # apply threshold to binarize the predictions
        if self.threshold:
            y_pred = tf.where(
                tf.less(y_pred, tf.zeros_like(y_pred) + self.threshold),
                tf.zeros_like(y_pred),
                tf.ones_like(y_pred)
            )

        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.cast(y_pred, tf.int32)
        # true positive
        true_positive = tf.math.count_nonzero(y_true * y_pred, 0)
        # predictions sum
        pred_sum = tf.math.count_nonzero(y_pred, 0)
        # true labels sum
        true_sum = tf.math.count_nonzero(y_true, 0)
        false_positive = pred_sum - true_positive
        false_negative = true_sum - true_positive
        y_true_negative = tf.math.not_equal(y_true, 1)
        y_pred_negative = tf.math.not_equal(y_pred, 1)
        true_negative = tf.math.count_nonzero(
            tf.math.logical_and(y_true_negative, y_pred_negative), axis=0
        )

        # true positive state update
        self.true_positives.assign_add(tf.cast(true_positive, self.dtype))
        # false positive state update
        self.false_positives.assign_add(tf.cast(false_positive, self.dtype))
        # false negative state update
        self.false_negatives.assign_add(tf.cast(false_negative, self.dtype))
        # true negative state update
        self.true_negatives.assign_add(tf.cast(true_negative, self.dtype))

    def result(self):
        flat_confusion_matrix = tf.convert_to_tensor(
            [
                self.true_negatives,
                self.false_positives,
                self.false_negatives,
                self.true_positives,
            ]
        )
        # reshape into 2*2 matrix
        confusion_matrix = tf.reshape(tf.transpose(flat_confusion_matrix), [-1, 2, 2])

        return confusion_matrix

    def get_config(self):
        """Returns the serializable config of the metric."""

        config = {
            "num_classes": self.num_classes,
        }
        base_config = super().get_config()
        return {**base_config, **config}

    def reset_state(self):
        reset_value = np.zeros(self.num_classes, dtype=np.int32)
        K.batch_set_value([(v, reset_value) for v in self.variables])

    def reset_states(self):
        # Backwards compatibility alias of `reset_state`. New classes should
        # only implement `reset_state`.
        # Required in Tensorflow < 2.5.0
        return self.reset_state()