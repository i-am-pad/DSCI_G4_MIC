import tensorflow as tf

class TrainParameters:
    '''configuration holder'''
    def __init__(self,
                 data_dir, save_dir, image_limit, image_size,
                 trial, epochs, batch_size, class_weight,
                 model, model_version, optimizer,
                 describe, verbose, debug,
                 **kwargs):
        ###############################
        # DATA
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.image_limit = image_limit
        self.image_size = image_size
        self.normalize = True
        
        ###############################
        # MODEL
        self.model = model
        self.model_version = model_version
        self.optimizer = optimizer
        
        # training
        self.trial = trial
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.train_size = 0.7
        self.validation_size = 0.2
        #self.test_size = 1. - self.train_size - self.validation_size
        
        self.learning_rate = 0.001
        self.weight_decay = 0.01
        
        # cnn
        # ...
        
        # ae
        # ...
        
        ###############################
        # HELP
        self.describe = describe
        self.verbose = verbose
        self.debug = debug

class InferParameters:
    '''configuration holder'''
    def __init__(self,
                 image_files, image_size, model_path, save_dir,
                 plot_layer_activations,
                 describe, verbose, debug,
                 **kwargs):
        ###############################
        # DATA
        self.image_files = image_files
        self.image_size = image_size
        self.model_path = model_path
        self.save_dir = save_dir
        self.normalize = True
             
        #######################
        # MODE
        self.plot_layer_activations = plot_layer_activations
        
        ###############################
        # HELP
        self.describe = describe
        self.verbose = verbose
        self.debug = debug