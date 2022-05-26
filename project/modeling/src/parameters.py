import tensorflow as tf

class Parameters:
    '''configuration holder'''
    def __init__(self,
                 data_dir, save_dir, image_limit, image_size, use_gpu,
                 trial, epochs, class_weight,
                 model, mode, optimizer,
                 describe, verbose,
                 **kwargs):
        ###############################
        # DATA
        
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.image_limit = image_limit
        self.image_size = image_size
        self.use_gpu = use_gpu and len(tf.config.list_physical_devices('GPU'))
        
        self.train_split = 0.7
        self.valid_split = 0.2
        self.normalize = True
                
        ###############################
        # MODEL
        self.model = model
        self.mode = mode
        self.optimizer = optimizer
        
        # training
        self.trial = trial
        self.epochs = epochs
        self.class_weight = class_weight
        
        self.batch_size = 32
        self.learning_rate = 0.001
        self.weight_decay = 0.01
        
        # cnn
        # ...
        
        ###############################
        # HELP
        self.describe = describe
        self.verbose = verbose