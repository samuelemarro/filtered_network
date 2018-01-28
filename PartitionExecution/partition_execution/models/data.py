import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import pickle
import tempfile
import shutil
import tarfile
import os
import numpy as np

class MNISTData:
    def __init__(self, path):
        mnist_dataset = input_data.read_data_sets(path, one_hot=True)
        self.train_dataset = tf.data.Dataset.from_tensor_slices((mnist_dataset.train.images, mnist_dataset.train.labels))
        self.validation_dataset = tf.data.Dataset.from_tensor_slices((mnist_dataset.validation.images, mnist_dataset.validation.labels))
        self.test_dataset = tf.data.Dataset.from_tensor_slices((mnist_dataset.test.images, mnist_dataset.test.labels))
        self.input_shape = [None, 784]
        self.output_shape = [None, 10]

        self.train_inputs = mnist_dataset.train.images
        self.train_labels = mnist_dataset.train.labels

        self.test_inputs = mnist_dataset.test.images
        self.test_labels = mnist_dataset.test.labels

        self.train_size = 55000
        self.test_size = 10000

class CIFAR10Data:

    def get_data(self, training, data_path):
        x = None
        y = None

        if training:
            for i in range(5):
                f = open(data_path + '/data_batch_' + str(i + 1), 'rb')
                datadict = pickle.load(f, encoding='latin1')
                f.close()

                _X = datadict['data']
                _Y = datadict['labels']

                _X = np.array(_X, dtype=float) / 255.0
                _X = _X.reshape([-1, 3, 32, 32])
                _X = _X.transpose([0, 2, 3, 1])
                _X = _X.reshape(-1, 32*32*3)

                if x is None:
                    x = _X
                    y = _Y
                else:
                    x = np.concatenate((x, _X), axis=0)
                    y = np.concatenate((y, _Y), axis=0)

        else:
            f = open(data_path + '/test_batch', 'rb')
            datadict = pickle.load(f, encoding='latin1')
            f.close()

            x = datadict['data']
            y = np.array(datadict['labels'])

            x = np.array(x, dtype=float) / 255.0
            x = x.reshape([-1, 3, 32, 32])
            x = x.transpose([0, 2, 3, 1])
            x = x.reshape(-1, 32*32*3)

        def dense_to_one_hot(labels_dense, num_classes=10):
            num_labels = labels_dense.shape[0]
            index_offset = np.arange(num_labels) * num_classes
            labels_one_hot = np.zeros((num_labels, num_classes))
            labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

            return labels_one_hot

        return x, dense_to_one_hot(y)

    def __init__(self, path):
        #maybe_download_and_extract()
        temp_dir = tempfile.mkdtemp()

        tar = tarfile.open(path)
        tar.extractall(temp_dir)
        tar.close()
            
        subdirectories = next(os.walk(os.path.join(temp_dir, '.')))[1]
        actual_path = os.path.join(temp_dir, subdirectories[0])

        train_x, train_y = self.get_data(True, actual_path)
        test_x, test_y = self.get_data(False, actual_path)
        shutil.rmtree(temp_dir)

        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_x, train_y))
        self.validation_dataset = None
        self.test_dataset = tf.data.Dataset.from_tensor_slices((test_x, test_y))

        self.input_shape = [None, 3072]
        self.output_shape = [None, 10]

        self.train_inputs = train_x
        self.train_labels = train_y

        self.test_inputs = test_x
        self.test_labels = test_y

        self.train_size = 50000
        self.test_size = 10000

