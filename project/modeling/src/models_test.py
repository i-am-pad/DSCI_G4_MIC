#!/usr/bin/env python

from collections import namedtuple
import os
# 0 = debug, 1 = info, 2 = warning, 3 = error
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import unittest

import dataset
import models.model
import models.cnn
import parameters

MockArgs = namedtuple('mock_args', 'data_dir save_dir image_limit image_size trial epochs batch_size class_weight model model_version optimizer describe verbose debug')

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
    
    def create_cnn_train_params():
        args = MockArgs(data_dir='gs://dsci591_g4mic/images_32x32',
                        save_dir='./data',
                        image_limit=200,
                        image_size=32,
                        trial='trial',
                        epochs=1,
                        batch_size=32,
                        class_weight=2,
                        model='cnn',
                        model_version='v1',
                        optimizer='adam',
                        describe=False,
                        verbose=True,
                        debug=False)
        return parameters.TrainParameters(**args._asdict())
    
    def create_cnn(self):
        model = models.model.get_model(self._cnn_train_params)
        return model
    
    def test_create_cnn(self):
        model = self.create_cnn()
        self.assertIsNotNone(model)
        self.assertEqual(type(model), models.cnn.CNN)
    
    def test_create_dataset(self):
        data_split = dataset.load_generators(self._cnn_train_params)
        self.assertIsNotNone(data_split)
        # can actually be less than this if there are fewer images than the limit for a given class
        total_expected = self._cnn_train_params.image_limit * len(dataset.G4MicDataGenerator.LABELS)
        self.assertEqual(sum([len(gen._filepaths) for gen in data_split.values()]), total_expected)
    
    def test_train_eval_cnn(self):
        model = self.create_cnn()
        data_split = dataset.load_generators(self._cnn_train_params)
        history = model.fit(data_split['train'],
                            validation_data = data_split['validation'],
                            epochs = self._cnn_train_params.epochs,
                            verbose = self._cnn_train_params.verbose,
                            )
        test_loss, test_acc, test_p, test_r, test_f1 = model.evaluate(data_split['test'], verbose=2 if self._cnn_train_params.verbose else 0)

if __name__ == '__main__':
    unittest.main()