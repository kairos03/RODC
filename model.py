# Copyright 2018 Kairos03. All Right Reserved.
"""
simple feature extraction model
"""

import tensorflow as tf


def make_model(inputs, labels, keep_prob, learning_rate):
    """
    make model
    """

    conv1 = tf.layers.conv2d(inputs, 64, (5, 5),
                             strides=(2, 2),
                             padding='same',
                             activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(conv1, (2, 2),
                                    strides=(2, 2),
                                    padding='same')

    conv2 = tf.layers.conv2d(pool1, 256, (1, 1),
                             strides=(1, 1),
                             padding='same',
                             activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv2, 512, (3, 3),
                             strides=(2, 2),
                             padding='same',
                             activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(conv3, (2, 2),
                                    strides=(2, 2),
                                    padding='same')


    feature_extraction = tf.identity(pool2, name='feature_extraction')

    reshape = tf.reshape(pool2, (-1, 20 * 20 * 512))

    dense1 = tf.layers.dense(reshape, 512, activation=tf.nn.relu)
    drop = tf.layers.dropout(dense1, rate=keep_prob)

    output = tf.layers.dense(drop, 2)

    with tf.name_scope('matrix'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=output, labels=labels), name='loss')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(output, 1), tf.argmax(labels, 1)), tf.float32), name='accuracy')
        optimizer = tf.train.AdamOptimizer(
            learning_rate, name="optimizer").minimize(loss)

    return output, loss, optimizer, accuracy
