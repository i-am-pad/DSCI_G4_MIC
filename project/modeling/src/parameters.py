import tensorflow as tf

class Parameters:
    '''configuration holder'''
    def __init__(self,
                 data_dir, save_dir, image_limit, image_size, use_gpu,
                 trial, epochs, save_model_train_data, progress_report_cadence,
                 model, mode,
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
                
        ###############################
        # MODEL
        self.model = model
        self.mode = mode
        
        # training
        self.trial = trial
        self.epochs = epochs
        self.save_model_train_data = save_model_train_data
        self.progress_report_cadence = progress_report_cadence
        
        self.batch_size = 32
        self.learning_rate = 0.001
        self.weight_decay = 0.01
        
        # CNN params
        # ...