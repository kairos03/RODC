# Copyright 2018 Kairos03. All Right Reserved.

from scipy import misc
import numpy as np
import tensorflow as tf


class Summary(object):
    @staticmethod
    def _summary(name, var):
        with tf.name_scope(name):
            _mean = tf.reduce_mean(var)
            _variance = tf.reduce_mean(tf.square(var - _mean))
            tf.summary.scalar('mean', _mean)
            tf.summary.scalar('variance', _variance)
            tf.summary.histogram(name, var)


class ConvLayer(Summary):
    """
    """

    def __init__(self, inputs, filters, k_size=(3, 3), strides=(1, 1), padding='SAME', name='conv2d'):
        """
        """
        self.inputs = inputs
        inputs_shape = inputs.get_shape().as_list()
        self.weight = [k_size[0], k_size[1], inputs_shape[-1], filters]
        self.bias = tf.constant(0.1, shape=[filters])
        self.strides = (1, strides[0], strides[1], 1)
        self.padding = padding
        self.name = name

    def out(self):
        """
        """
        with tf.variable_scope(self.name):

            conv = tf.nn.conv2d(self.inputs, tf.Variable(tf.truncated_normal(self.weight)),
                                self.strides, self.padding)
            act = tf.nn.relu(conv + tf.Variable(self.bias))
            self._summary('W', self.weight)
            self._summary('B', self.bias)
            print(act.shape)

            return act


def make_pool(inputs, k_size=(2, 2), strides=(2, 2), padding='SAME', name='pool'):
    with tf.name_scope(name):
        pool = tf.nn.max_pool(
            inputs, (1, k_size[0], k_size[1], 1), (1, strides[0], strides[1], 1), padding)

    print(pool.shape)

    return pool


class DenseLayer(Summary):

    def __init__(self, inputs, out_dim, name='dense'):
        self.inputs = inputs
        inputs_shape = inputs.get_shape().as_list()
        self.weight = [inputs_shape[-1], out_dim]
        self.bias = tf.constant(0.1, shape=[out_dim])
        self.name = name

    def out(self):
        with tf.variable_scope(self.name):
            dense = tf.matmul(self.inputs, tf.Variable(
                tf.truncated_normal(self.weight))) + tf.Variable(self.bias)
            act = tf.nn.relu(dense)
            self._summary('weight', self.weight)
            self._summary('bias', self.bias)
            print(act.shape)
            return act


def make_model(inputs, labels, keep_prob, learning_rate):

    tf.summary.image('input', inputs)

    # layers
    conv1 = ConvLayer(inputs, 64, k_size=(
        5, 5), strides=(2, 2), name='conv2d_1').out()
    pool1 = make_pool(conv1, name='pool_1')

    conv2 = ConvLayer(pool1, 192, strides=(2, 2), name='conv2d_2').out()
    pool2 = make_pool(conv2, name='pool_2')

    conv3 = ConvLayer(pool2, 128, k_size=(1, 1), name='conv2d_3').out()
    conv4 = ConvLayer(conv3, 256, name='conv2d_4').out()
    pool3 = make_pool(conv4, name='pool_3')

    conv5 = ConvLayer(pool3, 256, k_size=(1, 1), name='conv2d_5').out()
    conv6 = ConvLayer(conv5, 512, name='conv2d_6').out()
    pool4 = make_pool(conv6, name='pool_4')

    # conv7 = ConvLayer(pool4, 512, k_size=(1,1))
    # conv8 = ConvLayer(conv7, 1024)
    # pool5 = make_pool(conv8)

    reshape = tf.reshape(pool4, (-1, 5 * 5 * 512))
    print(reshape.shape)

    dense1 = DenseLayer(reshape, 300, name='dense_1').out()
    dropout = tf.nn.dropout(dense1, keep_prob)
    output = DenseLayer(dropout, 2, name='output').out()

    with tf.variable_scope('matrices'):
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=output, labels=labels), name='loss')

        optimizer = tf.train.AdamOptimizer(
            learning_rate, name='optimizer').minimize(loss)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(output, 1), tf.argmax(labels, 1)), tf.float32), name='accuracy')

    return output, loss, optimizer, accuracy
