#!/usr/bin/env python

from collections import namedtuple
import logging
import os
# 0 = debug, 1 = info, 2 = warning, 3 = error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import unittest

import dataset
import models.cnn
import models.logistic_regression
import models.model
import parameters

MockArgs = namedtuple('mock_args', 'data_dir save_dir image_limit image_size no_batch use_gpu trial epochs batch_size class_weight create_channel_dummies use_imagenet_weights dimension_reduction svc_l2 model model_version optimizer learning_rate weight_decay describe verbose debug')

class ModelsTestCase(unittest.TestCase):
    '''test cases for creating and training a model using data from GCS
    
    assumes google cloud utilities have been set up on the local desktop, and that the following environment variables are set:
        GOOGLE_APPLICATION_CREDENTIALS: see https://cloud.google.com/docs/authentication/getting-started, and use your own json key file
    
    alternatively, when running in colab, you can just authenticate using the following code:
        ```
        from google.colab import auth
        auth.authenticate_user()
        ```
    '''
    def setUp(self):
        tf.random.set_seed(42)
        self._cnn_train_params = ModelsTestCase.create_cnn_train_params()
        self._vgg16_train_params = ModelsTestCase.create_vgg16_train_params()
        self._vgg16_mpncov_train_params = ModelsTestCase.create_vgg16_mpncov_train_params()
        self._lr_params = ModelsTestCase.create_logistic_regression_params()
        self._svc_params = ModelsTestCase.create_svc_params()
    
    def create_cnn_train_params():
        args = MockArgs(data_dir='gs://dsci591_g4mic/images_32x32',
                        save_dir='./data',
                        image_limit=30,
                        image_size=32,
                        no_batch=False,
                        use_gpu=True,
                        trial='trial',
                        epochs=1,
                        batch_size=2,
                        create_channel_dummies=False,
                        use_imagenet_weights=False,
                        dimension_reduction=None,
                        svc_l2=0.01,
                        class_weight=2,
                        model='cnn',
                        model_version='cnn_v1',
                        optimizer='adam',
                        learning_rate=0.001,
                        weight_decay=0.0,
                        describe=False,
                        verbose=True,
                        debug=False)
        return parameters.TrainParameters(**args._asdict())
    
    def create_vgg16_train_params():
        args = MockArgs(data_dir='gs://dsci591_g4mic/images_32x32',
                        save_dir='./data',
                        image_limit=30,
                        image_size=32,
                        no_batch=False,
                        use_gpu=True,
                        trial='trial',
                        epochs=1,
                        batch_size=2,
                        class_weight=2,
                        create_channel_dummies=True,
                        use_imagenet_weights=None,
                        dimension_reduction=None,
                        svc_l2=0.01,
                        model='cnn',
                        model_version='vgg16_v1',
                        optimizer='adam',
                        learning_rate=0.001,
                        weight_decay=0.0,
                        describe=False,
                        verbose=True,
                        debug=False)
        return parameters.TrainParameters(**args._asdict())        
    
    def create_vgg16_mpncov_train_params():
        args = MockArgs(data_dir='gs://dsci591_g4mic/images_32x32',
                        save_dir='./data',
                        image_limit=30,
                        image_size=32,
                        no_batch=False,
                        use_gpu=True,
                        trial='trial',
                        epochs=1,
                        batch_size=2,
                        class_weight=2,
                        create_channel_dummies=True,
                        use_imagenet_weights=True,
                        dimension_reduction=64,
                        svc_l2=0.01,
                        model='cnn',
                        model_version='vgg16_mpncov_v1',
                        optimizer='adam',
                        learning_rate=0.001,
                        weight_decay=0.0,
                        describe=False,
                        verbose=True,
                        debug=False)
        return parameters.TrainParameters(**args._asdict())
    
    def create_logistic_regression_params():
        args = MockArgs(data_dir='gs://dsci591_g4mic/images_32x32',
                        image_size=32,
                        #data_dir=r'/mnt/d/data/dsci591_project/g4_mic_local/preprocessed/images_256x256',
                        #image_size=256,
                        save_dir='./data',
                        image_limit=30,
                        no_batch=True,
                        use_gpu=False,
                        trial='trial',
                        epochs=1,
                        batch_size=2,
                        class_weight=2,
                        create_channel_dummies=False,
                        use_imagenet_weights=True,
                        dimension_reduction=64,
                        svc_l2=0.01,
                        model='lr',
                        model_version='lr_v1',
                        optimizer='adam',
                        learning_rate=0.001,
                        weight_decay=0.0,
                        describe=False,
                        verbose=True,
                        debug=False)
        return parameters.TrainParameters(**args._asdict())
    
    def create_svc_params():
        args = MockArgs(#data_dir='gs://dsci591_g4mic/images_32x32',
                        #image_size=32,
                        data_dir=r'/mnt/d/data/dsci591_project/g4_mic_local/preprocessed/images_256x256',
                        image_size=256,
                        save_dir='./data',
                        image_limit=30,
                        no_batch=True,
                        use_gpu=False,
                        trial='trial',
                        epochs=1,
                        batch_size=2,
                        class_weight=2,
                        create_channel_dummies=False,
                        use_imagenet_weights=True,
                        dimension_reduction=64,
                        svc_l2=0.01,
                        model='lr',
                        model_version='svc_v1',
                        optimizer='adam',
                        learning_rate=0.001,
                        weight_decay=0.0,
                        describe=False,
                        verbose=True,
                        debug=False)
        return parameters.TrainParameters(**args._asdict())
    
    def create_cnn(self):
        model = models.model.get_model(self._cnn_train_params)
        return model
    
    def create_vgg16(self):
        model = models.model.get_model(self._vgg16_train_params)
        return model
    
    def create_vgg16_mpncov(self):
        model = models.model.get_model(self._vgg16_mpncov_train_params)
        return model
    
    def create_lr(self):
        model = models.model.get_model(self._lr_params)
        return model
        
    def create_svc(self):
        model = models.model.get_model(self._svc_params)
        return model
    
    def test_create_cnn(self):
        model = self.create_cnn()
        self.assertIsNotNone(model)
        self.assertEqual(type(model), models.cnn.CNN)
    
    def test_create_vgg16(self):
        model = self.create_vgg16()
        self.assertIsNotNone(model)
        self.assertEqual(type(model), models.cnn.VGG16)
    
    def test_create_vgg16_mpncov(self):
        model = self.create_vgg16_mpncov()
        self.assertIsNotNone(model)
        self.assertEqual(type(model), models.cnn.VGG16_MPNCOV)
    
    def test_create_lr(self):
        model = self.create_lr()
        self.assertIsNotNone(model)
        self.assertEqual(type(model), models.logistic_regression.LogisticRegression)
        
    def test_create_svc(self):
        model = self.create_svc()
        self.assertIsNotNone(model)
        self.assertEqual(type(model), models.logistic_regression.SVC)
    
    def test_create_dataset(self):
        data_split = dataset.load_generators(self._cnn_train_params)
        self.assertIsNotNone(data_split)
        # can actually be less than this if there are fewer images than the limit for a given class
        total_expected = self._cnn_train_params.image_limit * len(dataset.G4MicDataGenerator.LABELS)
        self.assertEqual(sum([len(gen._filepaths) for gen in data_split.values()]), total_expected)
    
    def test_create_dataset_with_channel_dummies(self):
        data_split = dataset.load_generators(self._vgg16_mpncov_train_params)
        self.assertIsNotNone(data_split)
        batch_images, batch_labels = data_split['train'][0]
        expected_channels = 3
        self.assertEqual(batch_images[0].shape[-1], expected_channels)
    
    def test_train_eval_cnn(self):
        model = self.create_cnn()
        data_split = dataset.load_generators(self._cnn_train_params)
        history = model.fit(data_split['train'],
                            validation_data = data_split['validation'],
                            epochs = self._cnn_train_params.epochs,
                            verbose = self._cnn_train_params.verbose,
                            shuffle=True,
                            )
        _ = model.evaluate(data_split['test'] if len(data_split['test']) else data_split['validation'], verbose=2 if self._cnn_train_params.verbose else 0)

    def test_train_eval_vgg16(self):
        model = self.create_vgg16()
        data_split = dataset.load_generators(self._vgg16_train_params)
        history = model.fit(data_split['train'],
                    validation_data = data_split['validation'],
                    epochs = self._vgg16_mpncov_train_params.epochs,
                    verbose = self._vgg16_mpncov_train_params.verbose,
                    shuffle=True,
                    )
        _ = model.evaluate(data_split['test'] if len(data_split['test']) else data_split['validation'], verbose=2 if self._vgg16_mpncov_train_params.verbose else 0)

    def test_train_eval_vgg16_mpncov(self):
        model = self.create_vgg16_mpncov()
        data_split = dataset.load_generators(self._vgg16_mpncov_train_params)
        history = model.fit(data_split['train'],
                            validation_data = data_split['validation'],
                            epochs = self._vgg16_mpncov_train_params.epochs,
                            verbose = self._vgg16_mpncov_train_params.verbose,
                            shuffle=True,
                            )
        _ = model.evaluate(data_split['test'] if len(data_split['test']) else data_split['validation'], verbose=2 if self._vgg16_mpncov_train_params.verbose else 0)
        
    def test_train_eval_lr(self):
        model = self.create_lr()
        data_split = dataset.load_generators(self._lr_params)
        history = model.fit(data_split['train'],
                            validation_data = data_split['validation'],
                            epochs = self._lr_params.epochs,
                            verbose = self._lr_params.verbose,
                            shuffle=True,
                            )
        _ = model.evaluate(data_split['test'] if len(data_split['test']) else data_split['validation'], verbose=2 if self._lr_params.verbose else 0)

    def test_train_eval_svc(self):
        model = self.create_svc()
        data_split = dataset.load_generators(self._svc_params)
        history = model.fit(data_split['train'],
                            validation_data = data_split['validation'],
                            epochs = self._svc_params.epochs,
                            verbose = self._svc_params.verbose,
                            shuffle=True,
                            )
        _ = model.evaluate(data_split['test'] if len(data_split['test']) else data_split['validation'], verbose=2 if self._svc_params.verbose else 0)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s.%(msecs)03d [%(levelname)s] %(module)s %(funcName)s: %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        )
    
    unittest.main()