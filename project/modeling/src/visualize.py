import datetime
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow.python.lib.io import file_io

import dataset
import models.model
import utilities

def plot_layer_activations(params, model, images_per_row=16):
    '''extremely helpful reference; one (main) chunk is verbatim!
    
    https://towardsdatascience.com/visualizing-intermediate-activations-of-a-cnn-trained-on-the-mnist-dataset-2c34426416c8
    '''
    
    model = models.model.load_model(params)
    data_split = dataset.load_generators(params)
    
    # all the conv / pooling layers only, which means this is super specific to cnn.get_model, and will need
    # to change to something more generic when we use this for other models next term
    layers = model.layers[1:6]
    am = tf.keras.Model(inputs=model.input, outputs=[l.output for l in layers])
    am_layer_names = [l.name for l in layers]
    
    image, _ = list(data_split)[0]

    save_path = os.path.join(params.save_dir,
                            f'{os.path.basename(params.model_dir)}_original_image.png')
    
    plt.imsave(file_io.FileIO(save_path, 'w'),
               image.numpy().reshape(params.image_size, params.image_size),
               vmin=0, vmax=255, cmap='gray')
    
    activations = am.predict(image)
    
    for layer_ix, (layer_name, activation) in enumerate(zip(am_layer_names, activations)):
        num_features = activation.shape[-1]
        size = activation.shape[1]
        
        num_cols = num_features // images_per_row
        display_grid = np.zeros((size * num_cols, size * images_per_row), dtype='float16')
        
        for c_ix in range(num_cols):
            for r_ix in range(images_per_row):
                #print(c_ix, r_ix, c_ix * images_per_row + r_ix, activation.shape)
                channel_image = activation[0, :, :, c_ix * images_per_row + r_ix]
                channel_image -= channel_image.mean()
                channel_image /= channel_image.std()
                channel_image *= 64
                channel_image += 128
                channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[c_ix*size:(c_ix + 1)*size,
                                r_ix*size:(r_ix + 1)*size
                                ] = channel_image
        
        scale = 1. / size
        
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        save_path = os.path.join(params.save_dir,
                                 f'{os.path.basename(params.model_dir)}_{layer_ix}_{layer_name}_activations.png')
        plt.imsave(file_io.FileIO(save_path, 'w'), display_grid, cmap='viridis')
        logging.info(f'saved layer ({layer_ix}, {layer_name}) activations to {save_path}')

def plot_model(params):
    '''plot the model's architecture
    
    https://keras.io/visualization/
    https://keras.io/api/utils/model_plotting_utils/
    '''
    model = models.model.get_model(params, compile=False)
    layers = model.layers
    am = tf.keras.Model(inputs=model.input, outputs=[l.output for l in layers])
    
    save_path = os.path.join(params.save_dir,
                            f'{os.path.basename(params.model_version)}.png')
    tf.keras.utils.plot_model(am,
                              to_file=save_path,
                              show_shapes=True,
                              show_dtype=True,
                              show_layer_names=True,
                              show_layer_activations=True,
                              expand_nested=True,
                              )
    logging.info(f'saved model architecture to {save_path}')

def show_model_summary(params):
    model = models.model.get_model(params, compile=False)
    utilities.summary_plus(model)


def print_confusion_matrix(confusion_matrix, axes, class_label, class_names, fontsize=14, do_plot=False, do_print=True):
    '''adapted from https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
    '''
    df_cm = pd.DataFrame(
        confusion_matrix.astype(np.int32), index=class_names, columns=class_names,
    )

    if do_print:
        logging.info(f'\n{class_label} Confusion Matrix\n{df_cm}\n')

    if do_plot:
        try:
            heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cbar=False, ax=axes)
        except ValueError:
            raise ValueError("Confusion matrix values must be integers.")
        heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=fontsize)
        heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=fontsize)
        axes.set_ylabel('y')
        axes.set_xlabel('y_pred')
        axes.set_title(f'{class_label} Confusion Matrix')

def plot_multilabel_confusion_matrix(confusion_matrices, labels, threshold, rows=5, cols=3, figsize=(12, 12), fontsize=14, save_chart=True, save_path=None):
    '''adapted from https://stackoverflow.com/questions/62722416/plot-confusion-matrix-for-multilabel-classifcation-python
    '''
    fig, ax = plt.subplots(rows, cols, figsize=figsize)
    
    for axes, cfs_matrix, label in zip(ax.flatten(), confusion_matrices, labels):
        print_confusion_matrix(cfs_matrix, axes, label, ["N", "Y"], fontsize=fontsize, do_plot=save_chart, do_print=not save_chart)
    
    fig.suptitle(f'Class Confusion Matrices (threshold={threshold})', fontsize=fontsize)
    fig.tight_layout()
    
    if save_chart and save_path:
        plt.savefig(save_path)
    elif not save_chart:
        plt.show()

def print_multilabel_confusion_matrix_singular(name, confusion_matrices, labels):
    class_names = ['N', 'Y']
    df_cms = []
    for cfs_matrix, label in zip(confusion_matrices, labels):
        df_cms.append(pd.DataFrame(
            cfs_matrix.astype(np.int32), index=pd.MultiIndex.from_arrays([[label] * 2, class_names]), columns=class_names,
        ))
    df_cm = pd.concat(df_cms)
    logging.info(f'\n{name} Confusion Matrix\n{df_cm}\n')

class MultiLabelConfusionMatrixPrintCallback(tf.keras.callbacks.Callback):
    def __init__(self, labels):
        super(MultiLabelConfusionMatrixPrintCallback, self).__init__()
        self._labels = labels
    
    def on_epoch_end(self, epoch, logs=None):
        if not 'multilabel_cm' in logs:
            return
        
        print_multilabel_confusion_matrix_singular(f'train {epoch}', logs['multilabel_cm'], self._labels)
        print_multilabel_confusion_matrix_singular(f'validation {epoch}', logs['val_multilabel_cm'], self._labels)
        
        return logs

class MultiLabelConfusionMatrixPlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, params, labels, save_dir=None):
        super(MultiLabelConfusionMatrixPlotCallback, self).__init__()
        self._params = params
        self._labels = labels
        self._save_dir = save_dir
    
    def on_epoch_end(self, logs=None):
        if not 'multilabel_cm' in logs:
            return
        
        model_detail = utilities.get_model_train_param_detail(self._params)
        time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = self._save_dir if self._save_dir else self._params.save_dir
        
        train_path = os.path.join(save_dir, f'{time}_train_confusion_matrix_{model_detail}.png')
        plot_multilabel_confusion_matrix(logs['multilabel_cm'], self._labels, self._params.threshold, save_chart=True, save_path=train_path)
        validation_path = os.path.join(save_dir, f'{time}_validation_confusion_matrix_{model_detail}.png')
        plot_multilabel_confusion_matrix(logs['val_multilabel_cm'], self._labels, self._params.threshold, save_chart=True, save_path=validation_path)
        
        logging.info(f'saved confusion matrix plots to {train_path} and {validation_path}')