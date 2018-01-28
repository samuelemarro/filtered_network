import tensorflow as tf

def cutoff_mask(output, cutoff):
    max_values = tf.reduce_max(output, axis=1)
    mask = tf.greater_equal(max_values, cutoff)
    mask.set_shape([None])
    #return mask
    #Di debug
    #return tf.concat([tf.ones([100], dtype=tf.bool), tf.zeros([100], dtype=tf.bool)], 0)
    #return tf.ones([200], dtype=tf.bool)
    #mask = tf.zeros_like(max_values, dtype=tf.bool)
    #mask.set_shape([None])
    return mask

def relative_mask(output, mask_parameter):
    max_values = tf.reduce_max(output, axis=1)
    values, _ = tf.nn.top_k(max_values, tf.to_int32(tf.multiply(tf.to_float(tf.shape(output)[0]), mask_parameter)))
    #Added to prevent errors when the size of values is 0
    values = tf.concat([[float('inf')], values], axis=0)
    bottom_value = values[-1]
    return tf.greater_equal(max_values, bottom_value)

def empty_mask(output, mask_parameter):
    max_values = tf.reduce_max(output, axis=1)
    return tf.zeros_like(max_values, dtype=tf.bool)

def full_mask(output, mask_parameter):
    max_values = tf.reduce_max(output, axis=1)
    return tf.ones_like(max_values, dtype=tf.bool)

def random_relative_mask(output, mask_parameter):
    return relative_mask(output, tf.squeeze(tf.random_uniform([1])))
