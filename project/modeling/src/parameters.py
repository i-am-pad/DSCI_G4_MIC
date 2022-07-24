import tensorflow as tf

class TrainParameters:
    '''configuration holder'''
    def __init__(self,
                 data_dir, save_dir, image_limit, image_size,
                 trial, epochs, batch_size, class_weight,
                 create_channel_dummies, use_imagenet_weights, dimension_reduction,
                 model, model_version, optimizer, learning_rate, weight_decay,
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
        
        # training
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.trial = trial
        self.epochs = epochs
        self.batch_size = batch_size
        self.class_weight = class_weight
        self.train_size = 0.8
        self.validation_size = 0.2
        #self.test_size = 1. - self.train_size - self.validation_size
        
        # cnn
        self.create_channel_dummies = create_channel_dummies
        self.use_imagenet_weights = use_imagenet_weights
        self.dimension_reduction = dimension_reduction
        
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
                 image_files, image_size, image_limit, model_dir, data_dir, save_dir,
                 mode,
                 model, model_version,
                 describe, verbose, debug,
                 **kwargs):
        ###############################
        # DATA
        self.image_files = image_files
        self.image_limit = image_limit
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.save_dir = save_dir
        self.normalize = True
             
        #######################
        # MODE
        self.mode = mode
        
        ###############################
        # MODEL
        self.model = model
        self.model_version = model_version
        
        ###############################
        # HELP
        self.describe = describe
        self.verbose = verbose
        self.debug = debug
        
        #########################
        # LEAKY ABSTRACTIONS
        self.batch_size = None
        self.image_size = 32
        self.optimizer = 'adam'
        self.use_imagenet_weights = False