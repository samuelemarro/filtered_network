from .. import partition_layer
import tensorflow as tf

class FeedForwardNetwork:
    def __init__(self, hidden_units, output_positions, optimizer, mask_maker=None, mask_parameters=None, hidden_activation=tf.nn.relu, output_activation=tf.identity):
        self.hidden_units = hidden_units
        self.output_positions = output_positions
        self.optimizer = optimizer
        self.mask_maker = mask_maker
        self.mask_parameters = mask_parameters
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

    def make_network(self, input, expected_output, use_cutoff, training):
        def train_op(output, labels, variables):
            with tf.variable_scope('cross_entropy'):
                diff = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=output)
                with tf.variable_scope('total'):
                    cross_entropy = tf.reduce_mean(diff)

            with tf.variable_scope('train'):
                train = self.optimizer.minimize(cross_entropy, var_list=variables)
            return train

        def accuracy_op(output, labels):
            with tf.variable_scope('accuracy'):
                with tf.variable_scope('correct_prediction'):
                    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(labels, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            return accuracy

        helper = partition_layer.PartitionHelper(use_cutoff, training, tf.shape(input)[0], expected_output.get_shape().as_list()[1], expected_output, train_op, dtype=input.dtype)

        output_size = expected_output.get_shape().as_list()[1]

        for i in range(len(self.output_positions)):
            previous_position = -1 if i == 0 else self.output_positions[i - 1]

            for j in range(previous_position + 1, self.output_positions[i] + 1):
                input = tf.layers.dense(input, self.hidden_units, activation=self.hidden_activation)
            
            last = i == len(self.output_positions) - 1

            output = None
            if helper.use_output(last):
                output = tf.layers.dense(input, output_size, activation=self.output_activation)

            mask = None
            if use_cutoff and not last:
                mask = self.mask_maker(output, self.mask_parameters[i])

            input = helper.add_checkpoint(input, output, last, mask=mask, train_maker=train_op)

        final_output = helper.reordered_output() if use_cutoff else output
        final_accuracy = None
        final_accuracy = accuracy_op(final_output, expected_output)
        train_ops = helper.train_ops
        variables = tf.trainable_variables()

        return final_output, final_accuracy, train_ops, variables, {}