import tensorflow as tf
import logging
import numpy
import time
import os

def preheat_network(session, tensor, feed_dict, preheat_count):
    for i in range(preheat_count):
        session.run(tensor, feed_dict=feed_dict)

def test_network(session, tensor, feed_dict, test_count):
    outputs = []
    start_time = time.time()
    for i in range(test_count):

        #run_meta = tf.RunMetadata()
        #session.run(tensor, feed_dict=feed_dict, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_meta)
        #tf.profiler.profile(session.graph, run_meta=run_meta, options=tf.profiler.ProfileOptionBuilder.time_and_memory())
        #os.system('pause')

        outputs.append(session.run(tensor, feed_dict=feed_dict))
        
    duration = time.time() - start_time
    return outputs, duration
    