import tensorflow as tf
import types
import os

from .. import partition

class FeedForwardNetwork:
    def get_complete_variable_scope(self):
        variable_scope = tf.get_variable_scope().name
        if variable_scope:
            variable_scope += '/'
        return variable_scope
    
    def hidden_maker(self, network, input, hidden_number):
        name = 'hidden_' + str(hidden_number)
        with tf.variable_scope(name):
            hidden = tf.layers.dense(input, self.hidden_units, activation=self.hidden_activation)
            variables = tf.global_variables(self.get_complete_variable_scope())
        if network.training:
            hidden = tf.nn.dropout(hidden, self.keep_prob, name='dropout')
        return hidden, variables

    def output_maker(self, network, hidden, output_number, output_size):
        name = 'output_' + str(output_number)
        with tf.variable_scope(name):
            output = tf.layers.dense(hidden, output_size, activation=self.output_activation)

            if not network.training:
                output = tf.nn.softmax(output)

            variables = tf.global_variables(self.get_complete_variable_scope())

        return output, variables

    def train_maker(self, network, output, labels, train_variables):
        with tf.variable_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output)
            with tf.variable_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)
        with tf.variable_scope('train'):
            train = self.optimizer.minimize(cross_entropy, var_list=train_variables)
        return train

    def accuracy_maker(self, network, output, labels):
        with tf.variable_scope('accuracy'):
            with tf.variable_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('train_accuracy', accuracy)
        return accuracy

    def __init__(self, hidden_units, optimizer=None, keep_prob=None, mask_maker=None, hidden_activation=tf.nn.relu, output_activation=tf.identity):
        self.hidden_units = hidden_units
        self.optimizer = optimizer
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        self.mask_maker = mask_maker

        self.keep_prob = keep_prob

class PostConvolutionalNetwork(FeedForwardNetwork):
    def hidden_maker(self, network, input, hidden_number):
        if hidden_number == self.convolution_position:
            if network.use_cutoff:
                filtered_input = tf.gather(network.input, network.empty_indices_slices[-1])
                #filtered_input = tf.Print(filtered_input, [tf.shape(filtered_input)[0]])

                filtered_input.set_shape([None, network.input.get_shape().as_list()[-1]])
            else:
                filtered_input = network.input

            convolution_output, convolution_variables = self.convolution(filtered_input, network)

            convolution_output = tf.reshape(convolution_output, [-1, convolution_output.get_shape()[1] * convolution_output.get_shape()[2] * convolution_output.get_shape()[3]])

            merged_input = tf.concat([convolution_output, input], 1)
            hidden_output, hidden_variables = super().hidden_maker(network, merged_input, hidden_number)
            variables = hidden_variables + convolution_variables

            return hidden_output, variables
        else:
            return super().hidden_maker(network, input, hidden_number)

        return super().output_maker(network, hidden, output_number, output_size)
    def __init__(self, hidden_units, convolution, convolution_position, optimizer = None, keep_prob = None, mask_maker = None, hidden_activation = tf.nn.relu, output_activation = tf.identity):
        self.convolution = convolution
        self.convolution_position = convolution_position
        return super().__init__(hidden_units, optimizer, keep_prob, mask_maker, hidden_activation, output_activation)

class ConvolutionalNetwork(FeedForwardNetwork):
    def hidden_maker(self, network, input, hidden_number):
        if hidden_number == 0:
            convolution_output, convolution_variables = self.convolution(input, network)
            
            convolution_output = tf.reshape(convolution_output, [-1, convolution_output.get_shape()[1] * convolution_output.get_shape()[2] * convolution_output.get_shape()[3]])
            print(convolution_output.get_shape())

            hidden_output, hidden_variables = super().hidden_maker(network, convolution_output, hidden_number)
            variables = hidden_variables + convolution_variables

            return hidden_output, variables
        else:
            return super().hidden_maker(network, input, hidden_number)

        return super().output_maker(network, hidden, output_number, output_size)
    def __init__(self, hidden_units, convolution, optimizer = None, keep_prob = None, hidden_activation = tf.nn.relu, output_activation = tf.identity):
        self.convolution = convolution
        return super().__init__(hidden_units, optimizer=optimizer, keep_prob=keep_prob, hidden_activation=hidden_activation, output_activation=output_activation)

class CustomFeedForwardNetwork(FeedForwardNetwork):
    def hidden_maker(self, network, input, hidden_number):
        if hidden_number in self.custom_layers.keys():
            return self.custom_layers[hidden_number](network, input, hidden_number, super().hidden_maker)
        else:
            return super().hidden_maker(network, input, hidden_number)

        return super().output_maker(network, hidden, output_number, output_size)
    def __init__(self, hidden_units, custom_layers, mask_maker = None, optimizer = None, keep_prob = None, hidden_activation = tf.nn.relu, output_activation = tf.identity):
        self.custom_layers = custom_layers
        return super().__init__(hidden_units, optimizer=optimizer, keep_prob=keep_prob, mask_maker=mask_maker, hidden_activation=hidden_activation, output_activation=output_activation)
