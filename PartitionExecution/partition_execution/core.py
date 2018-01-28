import tensorflow as tf
import logging
import time


def print_progress_bar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total: 
        print('')

def train_network(session, train_ops, train_dict, train_op_epochs=500, test_dict=None, print_accuracy=False, accuracy_ops=None, log_dir=None):
    save_logs = log_dir != None
    logger = logging.getLogger('partition')
    variable_values = dict()
    logger.info('Beginning training')

    graph = session.graph

    #Train the graph and save various summaries in the logs
    with graph.as_default():
        merged_summary = tf.summary.merge_all()
    
        with session.as_default():
            train_writer = None
            test_writer = None

            if save_logs:
                train_dir = log_dir + '/train'
                test_dir = log_dir + '/test'

                train_writer = tf.summary.FileWriter(train_dir, graph)
                test_writer = tf.summary.FileWriter(test_dir)

            session.run(tf.global_variables_initializer())
    
            total_training = len(train_ops) * train_op_epochs
            print_progress_bar(0, total_training, prefix = 'Progress:', suffix = 'Complete', length = 25)

            for train_number in range(len(train_ops)):
                for i in range(train_op_epochs):
                    summary_index = i + train_number * train_op_epochs
                    print_progress_bar(summary_index + 1, total_training, prefix = 'Progress:', suffix = 'Complete', length = 25)

                    #Train
                    session.run(train_ops[train_number], feed_dict=train_dict)

                    #Record test set summaries and accuracies
                    if i % 10 == 9:    
                        if save_logs:
                            summary = session.run(merged_summary, feed_dict=test_dict)
                            test_writer.add_summary(summary, summary_index)
                        if print_accuracy:
                            accuracy = session.run(accuracy_ops[train_number], feed_dict=test_dict)
                            logger.info('Accuracy at step %s (%s): %s' % (summary_index, train_number, accuracy))
            
                    #Record train set summaries
                    if save_logs:
                        save_metadata = i % 100 == 99

                        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE) if save_metadata else None
                        run_metadata = tf.RunMetadata() if save_metadata else None

                        summary = session.run(merged_summary, feed_dict=train_dict, options=run_options, run_metadata=run_metadata)
                        train_writer.add_summary(summary, summary_index)

                        if save_metadata:
                            train_writer.add_run_metadata(run_metadata, 'step {}-{}'.format(train_number, i))
                            logger.info('Adding run metadata for %s (%s)' % (summary_index, train_number))

            if save_logs:
                train_writer.close()
                test_writer.close()
                logger.info('Training finished!')
                logger.info('Training logs stored in {}'.format(train_dir))
                logger.info('Test logs stored in {}'.format(test_dir))
            logger.info('Saving variable values')


def compute_variable_dict(session, variables):
    variable_dict = dict()
    variable_values = session.run(variables)
    for i in range(len(variables)):
        variable_dict[variables[i].name] = variable_values[i]
    return variable_dict

def transfer_variables(session, inference_variables, variable_dict):
    logger = logging.getLogger('partition')

    #Copy the variables from the train graph to the inference graph
        
    graph = session.graph

    with graph.as_default():
        merged = tf.summary.merge_all()

        with session.as_default():
            session.run(tf.global_variables_initializer())

            #Load the variables from the train graph
            current_variables = dict()

            for variable in inference_variables:
                current_variables[variable.name] = variable

            logger.info('Loading variable values')
            logger.info('===============')
            variable_assignments = []

            for variable_name, variable_value in variable_dict.items():
                variable_assignments.append(tf.assign(current_variables[variable_name], variable_value))
                logger.info(variable_name)

            logger.info('===============')
            session.run(variable_assignments)

