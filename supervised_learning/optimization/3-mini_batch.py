#!/usr/bin/env python3
""" Train Mini-Batch"""

import tensorflow as tf
shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid, batch_size=32,
                     epochs=5, load_path="/tmp/model.ckpt",
                     save_path="/tmp/model.ckpt"):
    """ trains a loaded neural network model using mini-batch gradient descent

    Args:
        X_train (np.array): matrix of shape (m, 784) containing
        the training data
        Y_train (np.array): one-hot matrix of shape (m, 10)
        containing the training labels
        X_valid (np.array): matrix of shape (m, 784) containing
        the validation data
        Y_valid (np.array): one-hot matrix of shape (m, 10) containing the
        validation labels
        batch_size (int): number of data points in a batch
        epochs (int): number of times the training should pass through the
        whole dataset
        load_path (str): path from which to load the model
        save_path (str): path to where the model should be saved
    Returns:
        str: the path where the model was saved
    """
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph(load_path + '.meta')
        saver.restore(sess, load_path)
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        train_op = tf.get_collection('train_op')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        size = X_train.shape[0] // batch_size
        if X_train.shape[0] % batch_size != 0:
            size += 1
        for i in range(epochs + 1):
            cost_t, acc_t = sess.run([loss, accuracy],
                                     feed_dict={x: X_train, y: Y_train})
            cost_v, acc_v = sess.run([loss, accuracy],
                                     feed_dict={x: X_valid, y: Y_valid})
            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(cost_t))
            print("\tTraining Accuracy: {}".format(acc_t))
            print("\tValidation Cost: {}".format(cost_v))
            print("\tValidation Accuracy: {}".format(acc_v))
            if i < epochs:
                x_shuffled, y_shuffled = shuffle_data(X_train, Y_train)
                for i in range(size):
                    start = i * batch_size
                    end = start + batch_size
                    x_mini = x_shuffled[start:end]
                    y_mini = y_shuffled[start:end]
                    sess.run(train_op, feed_dict={x: x_mini, y: y_mini})
                    if (i + 1) % 100 == 0 and i > 0:
                        cost, acc = sess.run([loss, accuracy],
                                             feed_dict={x: x_mini, y: y_mini})
                        print("\tStep {}:".format(i + 1))
                        print("\t\tCost: {}".format(cost))
                        print("\t\tAccuracy: {}".format(acc))
        return saver.save(sess, save_path)
