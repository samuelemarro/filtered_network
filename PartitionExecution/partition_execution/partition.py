import tensorflow as tf
import abc
import logging
import numpy

def send_to_print(input):
    return tf.Print(tf.Print(input, [tf.shape(input)], '{} (Shape)'.format(input.name)), [input], '{} (Value)'.format(input.name))

class PartitionNetwork:

    def gather_output(self, gathered_indices, output, size):
        #Example: gathered_output = [C, A, B], gathered_indices = [2, 0, 1]

        #Get the positions of the indices (i.e. [1, 2, 0], which means "the first element (A)
        # is at index 1, the second (B) at index 2 and the third (C) at index 0"

        #Get the positions in descending order
        _, indices = tf.nn.top_k(gathered_indices, size, sorted=True)

        #Reverse the indices
        indices = tf.reverse(indices, [0])

        #Gather according to the indices
        return tf.gather(output, indices)

    def __init__(self, session, network_definition, use_cutoff, training, output_positions, input, expected_output, mask_parameters=None):
        self.use_cutoff = use_cutoff
        self.training = training
        self.output_positions = output_positions
        self.input = input
        self.expected_output = expected_output
        self.train_ops = []
        self.accuracy_ops = []
        self.output_layers = []
        self.masks = []
        self.empty_indices_slices = []
        self.train_variables = []
        self.final_output = None
        self.final_accuracy = None

        self.session = session

        hidden_layers = []
        masked_hiddens = []

        masked_labels = None

        with session.graph.as_default():
            output_compile_shape = expected_output.get_shape()
            output_runtime_shape = tf.shape(expected_output)

            if use_cutoff:
                with tf.variable_scope('filter_initialization'):
                    #This will contain the unordered outputs of the optimized network
                    gathered_output = tf.zeros([0, output_runtime_shape[-1]])

                    #This will contain the indices of the outputs of the optimized network. It
                    # is used to order the output to match `y_`.
                    gathered_indices = tf.zeros([0], dtype=tf.int32)
        
                    #Tracks which slots are empty
                    empty_indices = tf.range(output_runtime_shape[0])
                    self.empty_indices_slices.append(empty_indices)

            with tf.variable_scope('partitions'):
                for i in range(len(output_positions)):
                    with tf.variable_scope('partition_' + str(i)):
                        #These variables will be used by the training operator created in `train_maker`
                        partition_train_variables = []

                        with tf.variable_scope('hidden'):
                            #The first hidden layer of the first partition is connected to the input,
                            # while the first hidden layers of the other partitions are connected to the
                            # last hidden layers of the previous partitions.
                            if i == 0:
                                first_hidden_layer, first_hidden_variables = network_definition.hidden_maker(self, input, 0)
                            else:
                                first_hidden_layer, first_hidden_variables = network_definition.hidden_maker(self, masked_hiddens[-1] if use_cutoff else hidden_layers[-1], len(hidden_layers))
                        
                            hidden_layers.append(first_hidden_layer)
                            partition_train_variables += first_hidden_variables

                            #Create hidden layers until you add the layer that must be linked to the
                            # output layer
                            for j in range(1, output_positions[0] + 1) if i == 0 else range(output_positions[i - 1] + 1, output_positions[i]):
                                hidden_layer, hidden_variables = network_definition.hidden_maker(self, hidden_layers[-1], len(hidden_layers))
                                hidden_layers.append(hidden_layer)
                                partition_train_variables += hidden_variables
                        output_layer, output_variables = network_definition.output_maker(self, hidden_layers[-1], i, output_compile_shape[-1])
                    
                        self.output_layers.append(output_layer)

                        partition_train_variables += output_variables
                        self.train_variables += partition_train_variables

                        if use_cutoff:
                            if i != len(output_positions) - 1:
                                #If it's a normal partition, send some to the hidden and some
                                # to the output according to the cutoff rate

                                #The mask is false when you have to keep the result for more tests,
                                # while it's true when you can send it to output
                                with tf.variable_scope('mask'):
                                    mask = network_definition.mask_maker(output_layer, mask_parameters[i])
                                    self.masks.append(mask)
                                with tf.variable_scope('filtering'):
                                    masked_hidden = tf.boolean_mask(hidden_layers[-1], tf.logical_not(mask))
                                    masked_hiddens.append(masked_hidden)
                            with tf.variable_scope('gathering'):
                                if i == len(output_positions) - 1:
                                    masked_output = output_layer
                                    fill_indices = empty_indices
                                else:
                                    masked_output = tf.boolean_mask(output_layer, mask, name='output_gathering')
                                    fill_indices = tf.boolean_mask(empty_indices, mask, name='index_gathering')
                                gathered_output = tf.concat([gathered_output, masked_output], 0, name='output_update')
                                gathered_indices = tf.concat([gathered_indices, fill_indices], 0, name='filled_positions_update')

                                if i != len(output_positions) - 1:
                                    with tf.variable_scope('empty_positions_update'):
                                        empty_indices = tf.boolean_mask(empty_indices, tf.logical_not(mask))
                                        self.empty_indices_slices.append(empty_indices)
                        else:
                            self.train_ops.append(network_definition.train_maker(self, output_layer, expected_output, partition_train_variables))
                        self.accuracy_ops.append(network_definition.accuracy_maker(self, output_layer, expected_output))
        
            if use_cutoff:
                #Reorder `gathered_outputs` according to `gathered_indices`. We do not use tf.scatter_nd
                # because it is currently (Tensorflow v1.4.0) broken.
                with tf.variable_scope('post_processing'):
                   final_output = self.gather_output(gathered_indices, gathered_output, output_runtime_shape[0])
            
                self.final_output = tf.identity(final_output, 'final_output')
            else:
                self.final_output = tf.identity(self.output_layers[-1], 'final_output')
            self.final_accuracy = network_definition.accuracy_maker(self, self.final_output, expected_output)
