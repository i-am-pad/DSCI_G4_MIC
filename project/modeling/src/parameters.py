import tensorflow as tf

class DatasetParameters:
    def __init__(self,
                 data_dir, image_size, image_limit=None, batch_size=32, no_batch=False, use_gpu=False, create_channel_dummies=False,  multilabel=False,
                 train_size=0.8, validation_size=0.2, test_size=0.0,
                 verbose=False):
        self.data_dir = data_dir
        # not inferred from directory name, since not all datasets have this info
        # in their directory names
        self.image_size = image_size
        # loads image_limit of both benign and malicious images
        self.image_limit = image_limit
        self.batch_size = batch_size
        # == batch size of 1
        self.no_batch = no_batch
        # only relevant to models that require input with 3 channels
        self.create_channel_dummies = create_channel_dummies
        self.multilabel = multilabel
        
        self.train_size = train_size
        self.validation_size = validation_size
        self.test_size = test_size
        
        self.use_gpu = use_gpu
        self.verbose = verbose

class TrainParameters:
    '''configuration holder'''
    def __init__(self,
                 data_dir, save_dir, image_limit, image_size, crop_size, no_batch, use_gpu, workers, use_multiprocessing, max_queue_size,
                 trial, epochs, batch_size, dropout, multilabel,
                 create_channel_dummies, use_imagenet_weights, dimension_reduction,
                 svc_l2,
                 model, model_version, optimizer, learning_rate, weight_decay,
                 describe, verbose, debug,
                 **kwargs):
        ###############################
        # DATA
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.image_limit = image_limit
        self.image_size = image_size
        # PE headers generally don't go over a kilobyte in size
        self.crop_size = min(crop_size, 1024)
        self.no_batch = no_batch
        self.use_gpu = use_gpu
        self.workers = workers
        self.use_multiprocessing = use_multiprocessing
        self.max_queue_size = max_queue_size
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
        self.dropout_p = dropout
        self.multilabel = multilabel
        self.train_size = 0.8
        self.validation_size = 0.2
        #self.test_size = 1. - self.train_size - self.validation_size
        
        # cnn
        self.create_channel_dummies = create_channel_dummies
        self.use_imagenet_weights = use_imagenet_weights
        self.dimension_reduction = dimension_reduction
        
        # ae
        # ...
        
        # linear regression
        # ...
        
        # svc
        self.svc_l2 = svc_l2
        
        ###############################
        # HELP
        self.describe = describe
        self.verbose = verbose
        self.debug = debug

class InferParameters:
    '''configuration holder'''
    def __init__(self,
                 image_files, image_size, image_limit, model_dir, data_dir, save_dir, no_batch, use_gpu,
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
        self.no_batch = no_batch
        self.use_gpu = use_gpu
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