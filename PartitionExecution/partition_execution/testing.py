import tensorflow as tf
import logging
import numpy
import time
import os

def measure_flops(profile, root_op):
    def inputs(op):
        input_ops = frozenset([op])
        if len(op.inputs) > 0:
            for input in op.inputs:
                input_ops = input_ops.union(inputs(input.op))
        return input_ops

    def children(profile):
        if len(profile.children) > 0:
            children_profiles = dict()
            for child in profile.children:
                children_profiles.update(children(child))
            return children_profiles
        else:
            return {profile.name : profile.total_float_ops}

        
    nodes = frozenset(node.name for node in inputs(root_op))
    children_flops = children(profile)

    intersection = nodes.intersection(frozenset(children_flops.keys()))

    return numpy.sum(children_flops[name] for name in intersection)

def test_train_network(network, test_dict, test_count=1):
    train_accuracies = []
    train_flop_counts = []

    logger = logging.getLogger('partition')

    sess = network.session
    graph = sess.graph

    with graph.as_default():
        with sess.as_default():
            for i in range(len(network.accuracy_ops)):
                train_accuracy_sum = 0.0

                for j in range(test_count):
                    #Record accuracy and flop_count of the trained model

                    #Temporaneo
                    #if j % 10 == 0:
                    #    time.sleep(1)
    
                    train_accuracy = sess.run(network.accuracy_ops[i], feed_dict=test_dict)
                    logger.info('Training {} ({}) Accuracy: {:2.2f}%'.format(i + 1, j + 1, train_accuracy * 100.0))
                    train_accuracy_sum += train_accuracy
                   
                train_accuracies.append(train_accuracy_sum / float(test_count))

                train_flop_count_sum = 0.0

                for j in range(test_count):
                    run_metadata = tf.RunMetadata()
                    _ = sess.run(network.output_layers[i], feed_dict=test_dict, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)

                    options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.time_and_memory()).with_empty_output().build()
                    profile = tf.profiler.profile(graph, run_meta=run_metadata, cmd='scope', options=options)

                    #train_flop_count = measure_flops(profile, network.output_layers[i].op)
                    train_flop_count = profile.total_exec_micros
                    train_flop_count_sum += train_flop_count

                logger.info('Training {} Time: {}'.format(i + 1, train_flop_count_sum / test_count))
                logger.info('=============')

                train_flop_counts.append(train_flop_count_sum / test_count)


    return train_accuracies, train_flop_counts

def test_inference_network(network, inference_dict, test_count=1):
    logger = logging.getLogger('partition')

    sess = network.session
    graph = sess.graph

    with graph.as_default():
        with sess.as_default():
            inference_accuracy_sum = 0.0
            inference_flop_count_sum = 0.0
            mask_rates_sums = numpy.zeros([len(network.train_variables)])

            for i in range(test_count):
                #Temporaneo:
                #time.sleep(5)
                
                run_metadata = tf.RunMetadata()
                #Compute the accuracy
                inference_accuracy = sess.run(network.final_accuracy, feed_dict=inference_dict)
                logger.info('Accuracy: {:2.2f}%'.format(inference_accuracy * 100.0))

                _ = sess.run(network.final_output, feed_dict=inference_dict, options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)

                options = tf.profiler.ProfileOptionBuilder(tf.profiler.ProfileOptionBuilder.time_and_memory()).with_empty_output().build()
                profile = tf.profiler.profile(graph, run_meta=run_metadata, cmd='scope', options=options)
                inference_flop_count = profile.total_exec_micros #measure_flops(profile, network.final_output.op)

                logger.info('Inference FLOPS ({}): {}'.format(i + 1, inference_flop_count))

                final_output_size = sess.run(tf.shape(network.final_output), feed_dict=inference_dict)[0]
                logger.info('Final Output Size: {}'.format(final_output_size))
            

                #Store the size of the mask rates
                for j in range(len(network.masks)):
                    mask_size = sess.run(tf.shape(tf.where(network.masks[j])), feed_dict=inference_dict)[0]
                    mask_rate = mask_size / final_output_size
                    mask_rates_sums[j] += mask_rate

                    logger.info('Mask {} rate: {:2.2f}%'.format(j + 1, mask_rate * 100.0))

                inference_accuracy_sum += inference_accuracy
                inference_flop_count_sum += inference_flop_count
    
    return inference_accuracy_sum / float(test_count), inference_flop_count_sum / float(test_count), mask_rates_sums / float(test_count)

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
    