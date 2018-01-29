# Copyright 2018 Kairos03. All Right Reserved.
"""
simple feature extraction model
"""

import tensorflow as tf


def make_model(inputs, hparam):
    """
    make model
    """

    image = inputs['image']
    labels = inputs['labels']
    keep_prob = inputs['keep_prob']
    learning_rate = hparam['learning_rate']

    conv1 = tf.layers.conv2d(image, 64, (5, 5),
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

    conv4 = tf.layers.conv2d(pool2, 512, (1, 1),
                             strides=(1, 1),
                             padding='same',
                             activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(conv4, 1024, (3, 3),
                             strides=(1, 1),
                             padding='same',
                             activation=tf.nn.relu)
    pool3 = tf.layers.max_pooling2d(conv5, (2, 2),
                                    strides=(2, 2),
                                    padding='same')
    print(image.shape)
    print(pool1.shape)
    print(pool2.shape)
    print(pool3.shape)

    feature_extraction = tf.identity(pool3, name='feature_extraction')

    reshape = tf.reshape(pool3, (-1, 10 * 10 * 1024))

    dense1 = tf.layers.dense(reshape, 1024, activation=tf.nn.relu)
    drop = tf.layers.dropout(dense1, rate=keep_prob)

    output = tf.layers.dense(drop, 2)

    print(dense1.shape)
    print(output.shape)

    with tf.name_scope('matrix'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=output, labels=tf.stop_gradient(labels)), name='loss')
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(output, 1), tf.argmax(labels, 1)), tf.float32), name='accuracy')
        optimizer = tf.train.AdamOptimizer(
            learning_rate, name="optimizer").minimize(loss)

    return output, loss, optimizer, accuracy
