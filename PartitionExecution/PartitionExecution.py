import argparse
import os
import sys
import shutil
import time
import math

import numpy as np
import humanize
import logging

import tensorflow as tf

import partition_execution.partition as partition
import partition_execution.core as core
import partition_execution.testing as testing

import partition_execution.models.data as data
import partition_execution.models.network_definitions as networks
import partition_execution.models.masks as masks
from partition_execution.partition_layer import PartitionHelper
from tensorflow.python import debug as tf_debug

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d, avg_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

from matplotlib import pyplot as plot

def print_stats(collective_train_accuracies, collective_inference_accuracies, collective_train_flop_counts, collective_inference_flop_counts, collective_mask_rates):
    print('==============================')
    train_accuracies_averages = np.mean(collective_train_accuracies, axis=0)
    
    print('Train Accuracies:')
    for i in range(len(train_accuracies_averages)):
        print('\tAverage Train {} Accuracy: {:2.2f}%'.format(i + 1, train_accuracies_averages[i] * 100.0))
    
    mask_rates_averages = np.mean(collective_mask_rates, axis=0)

    print('Mask Rates:')
    for i in range(len(mask_rates_averages)):
        print('\tAverage Mask {} Rate: {:2.2f}%'.format(i + 1, mask_rates_averages[i] * 100.0))

    expected_accuracy = 0.0

    #Masked outputs
    for i in range(len(mask_rates_averages)):
        expected_accuracy += mask_rates_averages[i] * train_accuracies_averages[i]
    
    #Last output (no filtering)
    cumulative_mask_rate = np.sum(mask_rates_averages)
    expected_accuracy += (1 - cumulative_mask_rate) * train_accuracies_averages[-1]

    print('Expected Inference Accuracy: {:2.2f}%'.format(expected_accuracy * 100.0))

    average_inference_accuracy = np.mean(collective_inference_accuracies)
    print('Actual Inference Accuracy: {:2.2f}%'.format(average_inference_accuracy * 100.0))

    train_flop_count_averages = np.mean(collective_train_flop_counts, axis=0)
    average_inference_flop_count = np.mean(collective_inference_flop_counts)

    print('Training Graph FLOPS:')
    for i in range(len(train_flop_count_averages)):
        print('\tAverage Train {} FLOPS: {} ({})'.format(i + 1, train_flop_count_averages[i], humanize.intword(int(train_flop_count_averages[i]))))
    print('Average Inference FLOPS: {} ({})'.format(average_inference_flop_count, humanize.intword(int(average_inference_flop_count))))

    print('Inference Comparison:')
    for i in range(len(train_accuracies_averages)):
        print('\tCompared to Train {}:'.format(i + 1))
        print('\t\tFLOPS Difference: {:+2.2f}%'.format((average_inference_flop_count / train_flop_count_averages[i] - 1) * 100.0))
        print('\t\tAbsolute Accuracy Difference: {:+2.2f}%'.format((average_inference_accuracy - train_accuracies_averages[i]) * 100.0))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def compute_matches(outputs, labels, top_k=1):
    matches = []
    for output, label in zip(outputs, labels):
        indices = np.argpartition(output, -top_k)[-top_k:]
        matches.append(np.argmax(label) in indices)
    return np.array(matches)

def unpack(x):
    y = []
    for slice in x:
        for subslice in slice:
            y.append(subslice)
    return y

#Local Response Normalization refuses to accept an empty input, so we
#add an additional batch element and then remove it
def wrapped_local_response_normalization(input, use_cutoff):
    if use_cutoff:
        initial_shape = tf.shape(input)
        input = tf.concat([input, tf.zeros([1, initial_shape[1], initial_shape[2], initial_shape[3]])], axis=0)

    input = local_response_normalization(input)

    if use_cutoff:
        input = input[0:tf.maximum(initial_shape[0], 0)]

    return input

def foo(input, expected_output, use_cutoff, training):
    def train_op(output, labels, optimizer, variables):
        with tf.variable_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output)
            with tf.variable_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        #tf.summary.scalar('cross_entropy', cross_entropy)
        with tf.variable_scope('train'):
            train = optimizer.minimize(cross_entropy, var_list=variables)
        return train
    def accuracy_op(output, labels):
        with tf.variable_scope('accuracy'):
            with tf.variable_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        return accuracy

    optimizer = tf.train.AdamOptimizer()

    train_ops = []
    
    def standard_filtering(input, output, cutoff):
        helper.add_trainable_variables()
        if use_cutoff:
            if training:
                train_ops.append(train_op(output, expected_output, optimizer, helper.train_variables[-1]))
            else:
                mask = masks.cutoff_mask(output, cutoff)
                input = helper.apply_filter(input, output, mask)
        return input

    helper = PartitionHelper(tf.shape(input)[0], expected_output.get_shape().as_list()[1], dtype=input.dtype)

    input = tf.layers.dense(input, 200, activation=tf.nn.relu)
    input = tf.layers.dense(input, 200, activation=tf.nn.relu)

    output_1 = tf.layers.dense(input, 10, activation=tf.identity)

    input = standard_filtering(input, output_1, 0.5)

    input = tf.layers.dense(input, 200, activation=tf.nn.relu)
    input = tf.layers.dense(input, 200, activation=tf.nn.relu)
    output_2 = tf.layers.dense(input, 10, activation=tf.identity)

    helper.add_trainable_variables()

    if use_cutoff:
        helper.add_last_output(output_2)
    if training:
        if use_cutoff:
            train_ops.append(train_op(output_2, expected_output, optimizer, helper.train_variables[-1]))
        else:
            train_ops.append(train_op(output_2, expected_output, optimizer, tf.trainable_variables()))
    helper.add_trainable_variables()

    final_output = helper.reordered_output() if use_cutoff else output_2
    return final_output, accuracy_op(final_output, expected_output), train_ops, unpack(helper.train_variables), {}

#TODO:
##Devo per forza eliminare la cartella?
##Fare CIFAR-10
###Perché Optimised è più lento e accurato?
###Aggiungere variante FF -> Immagine ridotta -> Immagine completa?
##Forse posso rimuovere dei as_default()
##use_cutoff -> qualcosa
##Controllare i modelli e i parametri
##Rimuovere l'accuracy per il training?
#Promemoria:
#Se vuoi debuggare il filtering/gathering, attiva il train solo
#per la prima partition e imposta manualmente la mask
def main(_):
    base_accuracies = []
    base_times = []
    optimised_accuracies = []
    optimised_times = []
    cutoff_rates = [0.5]
    hidden_units = 150
    batch_size = 300
    train_count = 500
    hidden_layers = 4

    os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

    mnist_dir = 'C:/tmp/tensorflow/mnist/input_data'
    cifar10_path = 'C:/tmp/cifar-10/cifar-10-python.tar.gz'


    file_data = data.MNISTData(mnist_dir)
    #file_data = data.CIFAR10Data(cifar10_path)

    executions = 5

    logging.basicConfig(format='%(message)s', level=logging.DEBUG)

    logger = logging.getLogger('testing')

    def convolution(input, training):
        #input = tf.reshape(input, [-1, 24, 24, 3])
        #Temp
        input = tf.reshape(input, [-1, 32, 32, 3])

        def to_sparse(tensor, shape):
            indices = tf.where(tf.not_equal(input, 0))
            return tf.SparseTensor(indices, tf.gather_nd(tensor, indices), tensor.get_shape())


        repetitions = 12
        n = 160
        pooling_ratio = math.pow(2, 1 / 3)

        input_size = 2
        for i in range(repetitions):
            input_size = int(input_size * pooling_ratio)
            input_size += 1
        logger.info('Input Size: {}'.format(input_size))

        input_size = 94

        #input = to_sparse(input, [input.get_shape().as_list()[0], input_size, input_size, input.get_shape().as_list()[3]])

        padding = (input_size - input.get_shape().as_list()[1]) // 2

        input = tf.pad(input, [[0,0], [padding, padding], [padding, padding], [0,0]])

        logger.info('{}x{}'.format(input.get_shape().as_list()[1], input.get_shape().as_list()[1]))

        dropout_increase = 0.5 / (repetitions * 2 + 2)
        dropout_keep_prob = 0

        def apply_dropout(input, dropout_keep_prob):
            if training and dropout_keep_prob > 0:
                input = tf.nn.dropout(input, dropout_keep_prob)
            return input


        for i in range(repetitions):
            input = conv_2d(input, n * (i + 1), 2, padding='same', activation=tf.nn.leaky_relu, regularizer='L2')
            input = apply_dropout(input, dropout_keep_prob)
            dropout_keep_prob += dropout_increase

            input, _, _ = tf.nn.fractional_max_pool(input, [1, pooling_ratio, pooling_ratio, 1], pseudo_random=True)
            input = apply_dropout(input, dropout_keep_prob)
            dropout_keep_prob += dropout_increase

            logger.info('Downscaling image to {}'.format(input.get_shape().as_list()[1]))

        input = conv_2d(input, n * (repetitions + 1), 2, padding='same', activation=tf.nn.leaky_relu, regularizer='L2')
        input = apply_dropout(input, dropout_keep_prob)
        dropout_keep_prob += dropout_increase

        input = conv_2d(input, n * (repetitions + 2), 1, padding='same', activation=tf.nn.leaky_relu, regularizer='L2')
        input = apply_dropout(input, dropout_keep_prob)
        dropout_keep_prob += dropout_increase

        logger.info('Last Dropout: {}'.format(dropout_keep_prob))
        #input = conv_2d(input, 1, 1, activation='relu', regularizer='L2')


        #input = max_pool_2d(input, 2)

        #input = conv_2d(input, 64, 3, activation='relu', regularizer='L2')
        #input = max_pool_2d(input, 2)


        input = tf.reshape(input, [-1, input.get_shape()[1] * input.get_shape()[2] * input.get_shape()[3]])

        return input

    def standard_convolution(network, input, hidden_number, default_hidden):
        with tf.variable_scope('standard_convolution'):
            input = convolution(input, network.training)

            #input, _ = default_hidden(network, input, hidden_number)

            variable_scope = tf.get_variable_scope().name
            if variable_scope:
                variable_scope += '/'

            return input, tf.global_variables(variable_scope)

    def light_convolution(network, input, hidden_number, default_hidden):
        def central_crop(input, shape):
            x = input.get_shape()[1] // 2 - shape[0] // 2
            y = input.get_shape()[2] // 2 - shape[1] // 2
            return input[0:, x : x + shape[0], y : y + shape[1],0:]
        with tf.variable_scope('light_convolution'):
            input = tf.reshape(input, [-1, 32, 32, 3])

            input = avg_pool_2d(input, 2)

            input = tf.reduce_mean(input, axis=3, keep_dims=True)

            input = conv_2d(input, 32, 3, activation='relu', regularizer='L2')
            input = max_pool_2d(input, 2)

            input = conv_2d(input, 64, 3, activation='relu', regularizer='L2')
            input = max_pool_2d(input, 2)

            input = tf.reshape(input, [-1, input.get_shape()[1] * input.get_shape()[2] * input.get_shape()[3]])

            input, _ = default_hidden(network, input, hidden_number)

            variable_scope = tf.get_variable_scope().name
            if variable_scope:
                variable_scope += '/'

            return input, tf.global_variables(variable_scope)

    def filtered_convolution(network, input, hidden_number, default_hidden):
        with tf.variable_scope('filtered_convolution'):
            if network.use_cutoff:
                filtered_input = tf.gather(network.input, network.empty_indices_slices[-1])

                filtered_input.set_shape([None, network.input.get_shape().as_list()[-1]])
            else:
                filtered_input = network.input

            filtered_input = convolution(filtered_input, network.training)

            input = tf.concat([filtered_input, input], 1)

            input, _ = default_hidden(network, input, hidden_number)

            variable_scope = tf.get_variable_scope().name
            if variable_scope:
                variable_scope += '/'

            return input, tf.global_variables(variable_scope)

    def preprocessing(input, expected_output, use_cutoff, training):

        input = tf.reshape(input, [-1, 32, 32, 3])

        def central_crop(input, shape):
            x = input.get_shape()[1] // 2 - shape[0] // 2
            y = input.get_shape()[2] // 2 - shape[1] // 2
            return input[0:, x : x+shape[0], y : y + shape[1],0:]

        data_preprocessing = tflearn.DataPreprocessing()
        data_preprocessing.add_featurewise_zero_center()
        data_preprocessing.add_featurewise_stdnorm()

        data_augmentation = tflearn.ImageAugmentation()
        
        if training:
            data_augmentation.add_random_flip_leftright()
        #Temp
        #    input = tf.map_fn(lambda x: tf.random_crop(x, [24, 24, 3]), input)
        #else:
        #    input = central_crop(input, [24, 24])

        #Temp
        #input = tflearn.input_data([None, 24, 24, 3], input, data_preprocessing=data_preprocessing, data_augmentation=data_augmentation)
        input = tflearn.input_data([None, 32, 32, 3], input, data_preprocessing=data_preprocessing, data_augmentation=data_augmentation)

        logger.info(input.get_shape())

        input = tf.cast(input, dtype=tf.float32)

        #Temp
        #input = tf.reshape(input, [-1, 24 * 24 * 3])
        input = tf.reshape(input, [-1, 32 * 32 * 3])

        logger.info(input.get_shape())

        return input, expected_output

    base_custom_layers = {0 : standard_convolution}
    base_output_positions = [0]

    optimised_custom_layers = {5 : filtered_convolution}
    optimised_output_positions = [4, 5]

    a = tf.profiler.ProfileOptionBuilder().time_and_memory()

    for i in range(executions):
        #base_accuracy, base_time = run_base(False, file_data, batch_size, base_custom_layers, base_output_positions, preprocessing=preprocessing, keep_prob_value=0.9, train_epochs=train_count)
        base_accuracy, base_time = complete(False, file_data, batch_size, foo, train_op_epochs=train_count)

        base_accuracies.append(base_accuracy)
        base_times.append(base_time)

        #optimised_accuracy, optimised_time = run_base(True, file_data, batch_size, optimised_custom_layers, optimised_output_positions, preprocessing=preprocessing, mask_maker=masks.relative_mask, mask_parameters=cutoff_rates, keep_prob_value=0.9, train_epochs=train_count)
        optimised_accuracy, optimised_time = complete(True, file_data, batch_size, foo, train_op_epochs=train_count)

        optimised_accuracies.append(optimised_accuracy)
        optimised_times.append(optimised_time)

        
    logger.info('Mandatory sleeping to allow the GPU to cool off')
    time.sleep(60)
    
    final_base_accuracy = np.mean(base_accuracies)
    final_base_time = np.mean(base_times)
    final_optimised_accuracy = np.mean(optimised_accuracies)
    final_optimised_time = np.mean(optimised_times)

    print('Base Accuracy: {:2.2f}%'.format(final_base_accuracy * 100.0))
    print('Base Time: {} seconds'.format(final_base_time))
    print('Optimised Accuracy: {:2.2f}%'.format(final_optimised_accuracy * 100.0))
    print('Optimised Time: {} seconds'.format(final_optimised_time))
    print('{:f}x Speedup gained with an absolute {:2.2f}% accuracy difference'.format(final_base_time / final_optimised_time, (final_optimised_accuracy - final_base_accuracy) * 100.0))

    #print_stats(collective_train_accuracies, collective_inference_accuracies,
    #collective_train_flop_counts, collective_inference_flop_counts,
    #collective_mask_rates)

def dataset_sizes(data, batch_size):
    logger = logging.getLogger('testing')
    train_data_size = data.train_size
    train_data_size = data.train_size - (data.train_size % batch_size)

    test_data_size = data.test_size
    test_data_size = data.test_size - (data.test_size % batch_size)

    logger.info('Batch Size: {}'.format(batch_size))
    logger.info('Train Data Size: {}'.format(train_data_size))
    logger.info('Test Data Size: {}'.format(test_data_size))

    return train_data_size, test_data_size

def data_preprocessing(data, batch_size, train_session, inference_session):
    train_data_size, inference_data_size = dataset_sizes(data, batch_size)

    with train_session.as_default():
        with train_session.graph.as_default():
            train_iterator = data.train_dataset.take(train_data_size).shuffle(buffer_size=train_data_size).batch(batch_size).repeat().make_one_shot_iterator()
            train_input, train_expected_output = train_iterator.get_next()

            train_input.set_shape([batch_size, data.input_shape[1]])
            train_expected_output.set_shape([batch_size, data.output_shape[1]])

    with inference_session.as_default():
        with inference_session.graph.as_default():
            inference_iterator = data.test_dataset.take(test_data_size).batch(batch_size).repeat().make_one_shot_iterator()
            inference_input, inference_expected_output = inference_iterator.get_next()

            inference_input.set_shape([batch_size, data.input_shape[1]])
            inference_expected_output.set_shape([batch_size, data.output_shape[1]])

    return train_input, train_expected_output, inference_input, inference_expected_output, train_data_size, inference_data_size

def new_run_base(test_count, train_ops, final_output, train_variables, inference_variables, train_dict, inference_dict, train_session, inference_session, train_op_epochs=500):
    core.train_train_network(train_session, train_ops, train_dict, train_op_epochs)

    variable_dict = core.compute_variable_dict(train_session, train_variables)
    core.prepare_inference_network(inference_session, inference_variables, variable_dict)

    logger.info('Preheating...')
    testing.preheat_network(inference_session, final_output, None, test_count)
    base_outputs, execution_time = testing.test_network(inference_session, final_output, None, test_count)

    base_outputs = np.array(unpack(base_outputs))

    top1_matches = compute_matches(base_outputs, data.test_labels, top_k=1)
    top5_matches = compute_matches(base_outputs, data.test_labels, top_k=5)

    top1_accuracy = np.mean(top1_matches.astype(float))
    top5_accuracy = np.mean(top5_matches.astype(float))

    name = 'Optimised' if use_cutoff else 'Base'


    print('{} Top 1 Accuracy: {:2.2f}%'.format(name, top1_accuracy * 100.0))
    print('{} Top 5 Accuracy: {:2.2f}%'.format(name, top5_accuracy * 100.0))
    print('{} Time: {} seconds'.format(name, execution_time))

    return top1_accuracy, execution_time

def complete(use_cutoff, data, batch_size, network_maker, train_op_epochs=500):
    logger = logging.getLogger('testing')

    train_data_size, inference_data_size = dataset_sizes(data, batch_size)

    train_graph = tf.Graph()
    train_session = tf.Session(graph=train_graph)

    with train_session.as_default():
        with train_session.graph.as_default():
            train_iterator = data.train_dataset.take(train_data_size).shuffle(buffer_size=train_data_size).batch(batch_size).repeat().make_one_shot_iterator()
            train_input, train_expected_output = train_iterator.get_next()

            train_input.set_shape([batch_size, data.input_shape[1]])
            train_expected_output.set_shape([batch_size, data.output_shape[1]])

            _, train_accuracy, train_ops, train_variables, train_dict = network_maker(train_input, train_expected_output, use_cutoff, True)

    logger.info('{} Train Operations'.format(len(train_ops)))
    logger.info('===============')
    for i in range(len(train_ops)):
        logger.info('{}: {}'.format(i + 1, train_ops[i]))
    logger.info('===============')
    core.train_train_network(train_session, train_ops, train_dict, train_op_epochs)

    #a = train_session.run(train_accuracy)
    #print(a)
    #os.system('pause')

    inference_graph = tf.Graph()
    inference_session = tf.Session(graph=inference_graph)

    with inference_session.as_default():
        with inference_session.graph.as_default():
            inference_iterator = data.test_dataset.take(inference_data_size).batch(batch_size).repeat().make_one_shot_iterator()
            inference_input, inference_expected_output = inference_iterator.get_next()

            inference_input.set_shape([batch_size, data.input_shape[1]])
            inference_expected_output.set_shape([batch_size, data.output_shape[1]])

            inference_output, inference_accuracy, _, inference_variables, inference_dict = network_maker(inference_input, inference_expected_output, use_cutoff, False)

    test_count = inference_data_size // batch_size
    
    variable_dict = core.compute_variable_dict(train_session, train_variables)

    logger.info('Beginning inference')
    core.prepare_inference_network(inference_session, inference_variables, variable_dict)

    logger.info('Preheating...')
    testing.preheat_network(inference_session, inference_output, inference_dict, test_count)
    base_outputs, execution_time = testing.test_network(inference_session, inference_output, inference_dict, test_count)

    base_outputs = np.array(unpack(base_outputs))

    test_labels = np.asarray(data.test_labels)[0:inference_data_size, :]

    logger.info(base_outputs)
    logger.info(test_labels)

    top1_matches = compute_matches(base_outputs, test_labels, top_k=1)
    top5_matches = compute_matches(base_outputs, test_labels, top_k=5)

    top1_accuracy = np.mean(top1_matches.astype(float))
    top5_accuracy = np.mean(top5_matches.astype(float))

    name = 'Optimised' if use_cutoff else 'Base'


    print('{} Top 1 Accuracy: {:2.2f}%'.format(name, top1_accuracy * 100.0))
    print('{} Top 5 Accuracy: {:2.2f}%'.format(name, top5_accuracy * 100.0))
    print('{} Time: {} seconds'.format(name, execution_time))

    return top1_accuracy, execution_time

def run_base(use_cutoff, data, batch_size, custom_layers, output_positions, preprocessing=None, mask_maker=None, mask_parameters=None, hidden_units=100, keep_prob_value=0.9, train_epochs=500):
    logger = logging.getLogger('testing')

    def make_base_network(training, session, iterator, keep_prob=None):
        iterator_input, iterator_expected_output = iterator.get_next()

        iterator_input.set_shape([batch_size, data.input_shape[1]])
        iterator_expected_output.set_shape([batch_size, data.output_shape[1]])

        if preprocessing != None:
            iterator_input, iterator_expected_output = preprocessing(iterator_input, iterator_expected_output, use_cutoff, training)


        optimizer = tf.train.AdamOptimizer()

        network_definition = networks.CustomFeedForwardNetwork(hidden_units, custom_layers, mask_maker=mask_maker, optimizer=optimizer, keep_prob=keep_prob)

        print(network_definition.mask_maker)

        mp = tf.constant(mask_parameters) if (use_cutoff and not training) else None
        base_network = partition.PartitionNetwork(session, network_definition, use_cutoff and not training, training, output_positions, iterator_input, iterator_expected_output, mask_parameters=mp)

        return base_network

    train_data_size = data.train_size
    train_data_size = data.train_size - (data.train_size % batch_size)

    test_data_size = data.test_size
    test_data_size = data.test_size - (data.test_size % batch_size)

    logger.info('Batch Size: {}'.format(batch_size))
    logger.info('Train Data Size: {}'.format(train_data_size))
    logger.info('Test Data Size: {}'.format(test_data_size))

    train_graph = tf.Graph()
    train_session = tf.Session(graph=train_graph)

    with train_session.as_default():
        with train_graph.as_default():
            train_iterator = data.train_dataset.take(train_data_size).shuffle(buffer_size=train_data_size).batch(batch_size).repeat().make_one_shot_iterator()
            
            keep_prob = tf.placeholder(tf.float32, name='dropout_keep_probability')
            train_network = make_base_network(True, train_session, train_iterator, keep_prob=keep_prob)
            

    core.train_train_network(train_session, train_network.train_ops, {keep_prob : keep_prob_value}, train_op_epochs=train_epochs)

    inference_graph = tf.Graph()
    inference_session = tf.Session(graph=inference_graph)

    with inference_session.as_default():
        with inference_graph.as_default():
            inference_iterator = data.test_dataset.take(test_data_size).batch(batch_size).repeat().make_one_shot_iterator()
            inference_network = make_base_network(False, inference_session, inference_iterator)

    train_variables = train_network.train_variables
        
    inference_variables = inference_network.train_variables

    variable_dict = core.compute_variable_dict(train_session, train_variables)
    core.prepare_inference_network(inference_session, inference_variables, variable_dict)

    logger.info('Preheating...')
    testing.preheat_network(inference_session, inference_network.final_output, None, test_data_size // batch_size)
    base_outputs, execution_time = testing.test_network(inference_session, inference_network.final_output, None, test_data_size // batch_size)

    base_outputs = np.array(unpack(base_outputs))

    top1_matches = compute_matches(base_outputs, data.test_labels, top_k=1)
    top5_matches = compute_matches(base_outputs, data.test_labels, top_k=5)

    top1_accuracy = np.mean(top1_matches.astype(float))
    top5_accuracy = np.mean(top5_matches.astype(float))

    name = 'Optimised' if use_cutoff else 'Base'

    #train_writer = tf.summary.FileWriter(FLAGS.log_dir + '/' + name +
    #'/train', train_graph)
    #train_writer.flush()
    #train_writer.close()
    #inference_writer = tf.summary.FileWriter(FLAGS.log_dir + '/' + name +
    #'/inference', inference_graph)
    #inference_writer.flush()
    #inference_writer.close()
    #tf.train.write_graph(train_graph, FLAGS.log_dir + '/' + name + '/train',
    #'graph.pbtxt')
    #tf.train.write_graph(inference_graph, FLAGS.log_dir + '/' + name +
    #'/inference', 'graph.pbtxt')

    print('{} Top 1 Accuracy: {:2.2f}%'.format(name, top1_accuracy * 100.0))
    print('{} Top 5 Accuracy: {:2.2f}%'.format(name, top5_accuracy * 100.0))
    print('{} Time: {} seconds'.format(name, execution_time))

    return top1_accuracy, execution_time


def confidency_graph(output, expected_output, buckets=10):
    top1_matches = compute_matches(output, expected_output, top_k=1)
    top5_matches = compute_matches(output, expected_output, top_k=5)

    top1_outputs = output[np.nonzero(top1_matches)]
    top5_outputs = output[np.nonzero(np.logical_and(top5_matches, np.logical_not(top1_matches)))]
    wrong_outputs = output[np.nonzero(np.logical_not(np.logical_or(top1_matches, top1_matches)))]

    top1_confidencies = top1_outputs.max(axis=1)
    top5_confidencies = top5_outputs.max(axis=1)
    wrong_confidencies = wrong_outputs.max(axis=1)

    all_confidencies = np.concatenate((top1_confidencies, top5_confidencies, wrong_confidencies))

    top1_histogram, _ = np.histogram(top1_confidencies, buckets)
    all_histogram, _ = np.histogram(all_confidencies, buckets)

    top1_percentage = np.divide(top1_histogram, all_histogram)


    figure = plot.figure(1)
    plot.subplot(211)
    plot.bar(np.arange(buckets) + 1, top1_percentage)
    #plot.hist([wrong_confidencies, top5_confidencies, top1_confidencies],
    #20,(0, 1), histtype='barstacked')

    plot.show()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fake_data', nargs='?', const=True, type=bool,
                                            default=False,
                                            help='If true, uses fake data for unit testing.')
    parser.add_argument('--max_steps', type=int, default=250,
                                            help='Number of steps to run trainer.')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                                            help='Initial learning rate')
    parser.add_argument('--dropout', type=float, default=0.9,
                                            help='Keep probability for training dropout.')
    parser.add_argument('--data_dir',
            type=str,
            default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                                     'tensorflow/mnist/input_data'),
            help='Directory for storing input data')
    parser.add_argument('--log_dir',
            type=str,
            default=os.path.join(os.getenv('TEST_TMPDIR', '/tmp'),
                                                     'tensorflow/mnist/logs/mnist_with_summaries'),
            help='Summaries log directory')
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)

