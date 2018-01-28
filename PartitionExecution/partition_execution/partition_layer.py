import tensorflow as tf
import partition_execution.models.masks as masks

class PartitionHelper:
    def __init__(self, use_cutoff, training, output_batch_size, output_size, expected_output, train_maker, dtype=tf.float32):
        self.use_cutoff = use_cutoff
        self.training = training

        self.gathered_output = tf.zeros([0, output_size], dtype=dtype)
        self.tracking_indices = tf.range(output_batch_size, dtype=tf.int32)
        self.gathered_indices = tf.zeros([0], dtype=tf.int32)
        self.train_variables = []
        self.train_ops = []

        self.expected_output = expected_output
        self.train_maker = train_maker

    def apply_filter(self, hidden_tensor, output_tensor, mask):
        filtered_output = tf.boolean_mask(output_tensor, mask)
        filtered_indices = tf.boolean_mask(self.tracking_indices, mask)

        self.gathered_output = tf.concat([self.gathered_output, filtered_output], axis=0)
        self.gathered_indices = tf.concat([self.gathered_indices, filtered_indices], axis=0)

        negated_mask = tf.logical_not(mask)

        self.tracking_indices = tf.boolean_mask(self.tracking_indices, negated_mask)

        return tf.boolean_mask(hidden_tensor, negated_mask)

    def add_last_output(self, output_tensor):
        self.gathered_output = tf.concat([self.gathered_output, output_tensor], axis=0)
        self.gathered_indices = tf.concat([self.gathered_indices, self.tracking_indices], axis=0)
        self.tracking_indices = []

    def current_scope(self):
        variable_scope = tf.get_variable_scope().name
        if variable_scope:
            variable_scope += '/'
        return variable_scope

    def add_trainable_variables(self, scope=None):
        variables = tf.trainable_variables(scope) #Invece di global_variables?

        for variable_collection in self.train_variables:
            variables = [x for x in variables if x not in variable_collection]

        self.train_variables.append(variables)
    
    def fake_scatter_nd(self, params, indices):
        #Example: params = [C, A, B], indices = [2, 0, 1]

        #Get the positions of the indices (i.e. [1, 2, 0], which means "the first element (A)
        # is at index 1, the second (B) at index 2 and the third (C) at index 0"

        #Get the positions in descending order
        _, indices = tf.nn.top_k(indices, tf.shape(indices)[0], sorted=True)

        #Reverse the indices
        indices = tf.reverse(indices, [0])

        #Gather according to the indices
        return tf.gather(params, indices)

    def use_output(self, last):
        return self.use_cutoff or last

    def add_checkpoint(self, input, output, last, mask=None):
        if self.use_cutoff:
            self.add_trainable_variables()
            if last:
                self.add_last_output(output)

            if self.training:
                self.train_ops.append(self.train_maker(output, self.expected_output, self.train_variables[-1]))
            elif not last:
                input = self.apply_filter(input, output, mask)
        elif last and self.training:
            all_variables = tf.trainable_variables()

            self.train_ops.append(self.train_maker(output, self.expected_output, all_variables))
            self.train_variables.append(all_variables)

        return input

    def reordered_output(self):
        return self.fake_scatter_nd(self.gathered_output, self.gathered_indices)