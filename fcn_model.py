# Copyright kairos03 2018. All Right Reserved.
"""
fcn_model
"""

import tensorflow as tf
import numpy as np
from scipy import misc
from data import process
import matplotlib.pyplot as plt

n_classes = 3
input_shape = (256, 256)
label_shape = (256, 256)
deconv_weight = 2


def pre_process(image_name):
    x = []
    y = []
    for name in image_name:
        xim = misc.imread(process.ORIGIN_PATH + name)
        yim = misc.imread(process.ANNO_PATH + name)

        xim = misc.imresize(xim, input_shape)
        yim = misc.imresize(yim, label_shape)

        xim = np.true_divide(xim, 255.)

        x.append(xim)
        y.append(yim)

    x = np.stack(x)
    y = np.stack(y)

    return x, y


def dilated_conv2d(x, filters, ksize=[3, 3], rate=2, padding='SAME'):
    """
    """
    filters = tf.Variable(tf.truncated_normal(
        ksize + x.get_shape()[-1] + [filters]))
    baises = tf.Variable(tf.constant(0.1, tf.float16))

    x = tf.nn.batch_normalization(x)
    conv = tf.nn.atrous_conv2d(x, filters, rate, padding)
    act = tf.nn.relu(conv)

    return act


def conv2d(x, filters, ksize=[3, 3], strides=[1, 1], padding='SAME'):
    """
    """
    with tf.name_scope('conv2d'):
        xshape = x.shape.as_list()
        # biases = tf.Variable(tf.constant(0.1, shape=[filters]))
        filters = tf.Variable(tf.truncated_normal(
            ksize + [xshape[-1]] + [filters]))
        strides = [1] + strides + [1]

        x = tf.nn.batch_normalization(x, 0, 1, None, None, 1e-4)
        conv = tf.nn.conv2d(x, filters, strides, padding)
        # add = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(conv)

        tf.summary.scalar('mean', tf.reduce_mean(filters))

    return act


def deconv2d(x, filters, output_shape, ksize=[3, 3], strides=[1, 1], padding='SAME'):
    """
    """
    with tf.name_scope('deconv2d'):

        xshape = x.shape.as_list()
        # biases = tf.Variable(tf.constant(0.1, shape=[filters]))
        filters = tf.Variable(tf.truncated_normal(
            ksize + [filters] + [xshape[-1]]))
        strides = [1] + strides + [1]

        x = tf.nn.batch_normalization(x, 0, 1, None, None, 1e-4)
        conv = tf.nn.conv2d_transpose(
            x, filters, output_shape, strides, padding)
        # add = tf.nn.bias_add(conv, biases)
        act = tf.nn.relu(conv)

        tf.summary.scalar('mean', tf.reduce_mean(filters))

    return act


def max_pool(x, ksize=[2, 2], strides=[2, 2], padding='SAME'):
    ksize = [1] + ksize + [1]
    strides = [1] + strides + [1]
    return tf.nn.avg_pool(x, ksize, strides, padding)


def make_model(inputs, labels, keep_prob, learning_rate):

    tf.summary.image('input', inputs)
    tf.summary.image('label', labels)

    # conv
    with tf.name_scope('conv1'):
        conv1_1 = conv2d(inputs, 64)
        conv1_2 = conv2d(conv1_1, 64, ksize=[5, 5])
        pool1 = max_pool(conv1_2)

    with tf.name_scope('conv2'):
        conv2_1 = conv2d(pool1, 128)
        conv2_2 = conv2d(conv2_1, 128, ksize=[5, 5])
        pool2 = max_pool(conv2_2)

    with tf.name_scope('conv3'):
        conv3_1 = conv2d(pool2, 256, ksize=[1, 1])
        conv3_2 = conv2d(conv3_1, 256)
        conv3_3 = conv2d(conv3_2, 256)
        pool3 = max_pool(conv3_3)

    with tf.name_scope('conv4'):
        conv4_1 = conv2d(pool3, 512, ksize=[1, 1])
        conv4_2 = conv2d(conv4_1, 512)
        conv4_3 = conv2d(conv4_2, 512)
        pool4 = max_pool(conv4_3)

    with tf.name_scope('fcn'):
        conv5 = conv2d(pool4, 1024, ksize=[1, 1])
        drop1 = tf.nn.dropout(conv5, keep_prob)
        conv6 = conv2d(drop1, 1024, ksize=[1, 1])
        drop2 = tf.nn.dropout(conv6, keep_prob)
        conv7 = conv2d(drop2, 3, ksize=[1, 1])

    # deconv
    deconv1 = deconv2d(
        conv7, 256, output_shape=tf.shape(pool3),  strides=[2, 2])
    fuse1 = tf.add(deconv1, deconv_weight * pool3)

    deconv2 = deconv2d(
        fuse1, 128, output_shape=tf.shape(pool2), strides=[2, 2])
    fuse2 = tf.add(deconv2, deconv_weight * pool2)

    deconv3 = deconv2d(fuse2, 64, output_shape=tf.shape(pool1), strides=[2, 2])
    fuse3 = tf.add(deconv3, deconv_weight * pool1)

    shape = tf.shape(inputs)
    out_sh = tf.stack([shape[0], shape[1], shape[2], n_classes])
    output = deconv2d(fuse3, n_classes, ksize=[
                      5, 5], output_shape=out_sh, strides=[2, 2])

    tf.summary.image('output', output)

    print('input', inputs.shape)
    print('pool1', pool1.shape)
    print('pool2', pool2.shape)
    print('pool3', pool3.shape)
    print('pool4', pool4.shape)
    print('conv7', conv7.shape)
    print('fuse1', fuse1.shape)
    print('fuse2', fuse2.shape)
    print('fuse3', fuse3.shape)
    print('output', output.shape)

    with tf.name_scope('matrix'):

        output = tf.reshape(output, (-1, n_classes))
        labels = tf.reshape(labels, (-1, n_classes))

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            logits=output, labels=tf.stop_gradient(labels)), name='loss')
        optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return output, loss, optimizer


if __name__ == '__main__':
    names = ['0000001.jpg', '0000002.jpg', '0000003.jpg', '0000004.jpg']
    x, y = pre_process(names)
    print(y[0])
    plt.imshow(y[0])
    plt.show()
