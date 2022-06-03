import tensorflow as tf

class TrainParameters:
    '''configuration holder'''
    def __init__(self,
                 data_dir, save_dir, image_limit, image_size, use_gpu,
                 trial, epochs, batch_size, class_weight, save_model_evaluation,
                 model, optimizer,
                 describe, verbose, debug,
                 **kwargs):
        ###############################
        # DATA
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.image_limit = image_limit
        self.image_size = image_size
        self.use_gpu = use_gpu and len(tf.config.list_physical_devices('GPU'))
        self.normalize = True
                
        ###############################
        # MODEL
        self.model = model
        self.optimizer = optimizer
        
        # training
        self.trial = trial
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.save_model_evaluation = save_model_evaluation
        
        self.learning_rate = 0.001
        self.weight_decay = 0.01
        
        # cnn
        # ...
        
        ###############################
        # HELP
        self.describe = describe
        self.verbose = verbose
        self.debug = debug

class InferParameters:
    '''configuration holder'''
    def __init__(self,
                 image_files, image_size, model_dir, save_dir, use_gpu,
                 plot_layer_activations,
                 describe, verbose, debug,
                 **kwargs):
        ###############################
        # DATA
        
        self.image_files = image_files
        self.image_size = image_size
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.use_gpu = use_gpu and len(tf.config.list_physical_devices('GPU'))
        
        self.normalize = True
             
        #######################
        # MODE
        self.plot_layer_activations = plot_layer_activations
        
        ###############################
        # HELP
        self.describe = describe
        self.verbose = verbose
        self.debug = debug