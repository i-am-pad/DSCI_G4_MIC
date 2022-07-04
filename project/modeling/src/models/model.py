import tensorflow as tf
import tensorflow_addons as tfa

from . import cnn

MODEL_MODULES = {
    'cnn': cnn,
}

def get_model(params) -> tf.keras.Model:
    '''returns an untrained model based on the model_version parameter
    
    args
        params.model: model to create
        params.model_version: version of the model to create
    
    returns
        instance of untrained model
    
    raises
        ValueError if the model or model_version is not supported
    '''
    model_module = MODEL_MODULES.get(params.model)
    if not model_module:
        raise ValueError(f'unknown model: {params.model}')
    
    model_version = model_module.MODEL_VERSION.get(params.model_version)
    if not model_version:
        raise ValueError(f'unknown model version: {params.model_version}')
    
    return model_version(params)

def load_model(params, has_f1: True) -> tf.keras.Model:
    '''loads a model from a tensorflow format save directory
    
    args
        params.model_path: path to the h5 or tf directory of a model checkpoint to load
        has_f1: enables loading external library's F1 score metric
    
    returns
        target model trained to the point of when the checkpoint was saved
    '''
    custom_objects = {}
    
    if has_f1:
        def f1(y_true, y_pred):
            '''wrapper to help loading keras models saved with tensorflow_addons F1Score metric
            '''
            metric = tfa.metrics.F1Score(num_classes=3, threshold=0.5)
            metric.update_state(y_true, y_pred)
            return metric.result()
        custom_objects['f1'] = f1
    
    return tf.keras.models.load_model(params.model_path.rstrip('/'), compile=False, custom_objects=custom_objects)