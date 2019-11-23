
import argparse
import logging
import math
import tempfile
import time

import tensorflow as tf
import nni
from lib.model.unet import Unet
from lib.dataset.celebAMaskHQ_tf import get_tf_dataset

FLAGS = None

logger = logging.getLogger('CelebAMaskHQ_AutoML')



def main(params):
    '''
    Main function, build mnist network, run and send result to NNI.
    '''
    # Import data

    train_dataset = get_tf_dataset(tf_folder, policy, shuffle=True)
    print('Mnist download data done.')
    logger.debug('Mnist download data done.')

    # Create the model
    # Build the graph for the deep net
    mnist_network = MnistNetwork(channel_1_num=params['channel_1_num'],
                                 channel_2_num=params['channel_2_num'],
                                 conv_size=params['conv_size'],
                                 hidden_size=params['hidden_size'],
                                 pool_size=params['pool_size'],
                                 learning_rate=params['learning_rate'])
    mnist_network.build_network()
    logger.debug('Mnist build network done.')

    # Write log
    graph_location = tempfile.mkdtemp()
    logger.debug('Saving graph to: %s', graph_location)
    train_writer = tf.summary.FileWriter(graph_location)
    train_writer.add_graph(tf.get_default_graph())

    test_acc = 0.0
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(params['batch_num']):
            batch = mnist.train.next_batch(params['batch_size'])
            mnist_network.train_step.run(feed_dict={mnist_network.images: batch[0],
                                                    mnist_network.labels: batch[1],
                                                    mnist_network.keep_prob: 1 - params['dropout_rate']}
                                        )

            if i % 100 == 0:
                test_acc = mnist_network.accuracy.eval(
                    feed_dict={mnist_network.images: mnist.test.images,
                               mnist_network.labels: mnist.test.labels,
                               mnist_network.keep_prob: 1.0})

                nni.report_intermediate_result(test_acc)
                logger.debug('test accuracy %g', test_acc)
                logger.debug('Pipe send intermediate result done.')

        test_acc = mnist_network.accuracy.eval(
            feed_dict={mnist_network.images: mnist.test.images,
                       mnist_network.labels: mnist.test.labels,
                       mnist_network.keep_prob: 1.0})

        nni.report_final_result(test_acc)
        logger.debug('Final result is %g', test_acc)
        logger.debug('Send final result done.')


def get_params():
    ''' Get parameters from command line '''
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default='/tmp/tensorflow/mnist/input_data', help="data directory")
    parser.add_argument("--dropout_rate", type=float, default=0.5, help="dropout rate")
    parser.add_argument("--channel_1_num", type=int, default=32)
    parser.add_argument("--channel_2_num", type=int, default=64)
    parser.add_argument("--conv_size", type=int, default=5)
    parser.add_argument("--pool_size", type=int, default=2)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--batch_num", type=int, default=2700)
    parser.add_argument("--batch_size", type=int, default=32)

    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    try:
        # get parameters form tuner
        tuner_params = nni.get_next_parameter()
        logger.debug(tuner_params)
        tuner_params['batch_num'] = tuner_params['TRIAL_BUDGET'] * 100
        params = vars(get_params())
        params.update(tuner_params)
        main(params)
    except Exception as exception:
        logger.exception(exception)
        raise
