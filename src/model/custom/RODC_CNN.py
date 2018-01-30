# copyright 2018 quisutdeus7 all reserved
# implement : https://arxiv.org/pdf/1506.02640.pdf
import tensorflow as tf
import os
import numpy as np

# model ingredients
# parameter
LR = 1e-3

# tensorboard
'''
def var_summary(var):
    """ weight variable summary
    """
    with tf.name_scope('summary'):
        tf.summary.histogram('histogram', var)
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
'''

# leaky Relu function
def lrelu(x, alpa = 0.1):
    return tf.maximum(x, alpa*x)

# convolution 2d layer
def conv2d(net, filters, name, activation = 'Relu', k_str = [1,3,3,1], _padding='SAME'):
    layer = tf.nn.conv2d(net, filters, k_str, 'SAME', name=name, padding = _padding)
    return layer

# batch normalization
def batch_norm(net, center = True, scale = True, epsilon = 1e-3, training = False):
    net = tf.layers.batch_normalization(net, center=center, scale=scale, epsilon=epsilon, trainable=training)
    
    if not center:
        b = tf.Variable(tf.zeros(int(net.shape[3]), name='bias'))
        net = tf.nn.bias_add(net, b, name='add bias')
    return net

# max pool layer
def max_pool(net, k_size = [2,2], k_str = [1,2,2,1], name=None):
    return tf.nn.max_pool(net, ksize=k_size, strides=k_str, padding='SAME', name=name)
# ~ # model ingredients

'''
# model
# modify YOLO architect(448*448 -> 200*200, 7*7*20 ->5*5*)
def darknet(net, classes, num_anchors, training = False, center = True):
    layer = conv2d(net, 32, name = 'RODC_conv1')
    layer = batch_norm(layer)
    layer = max_pool(layer, name='RODC_max1')
'''
# This must transfer to train.py
def next_batch(batch, trainable=True, one_hot=True):
    if trainable:
        # tensor f_img, s_img shpae : width, height, depth, total num
        xs = tf.concat([f_img, s_img], 2)
        xs = xs[:][:][:][batch:(batch+1)*batch_size]
        # indicate elements by 0,1
        ys = "?"
    else:
        # ALL img
        xs =xs[:][:][:][:]
        yx = "?"
        
    # width, height, depth, total num --> num, w, h, d 
    xs = tf.transpose(xs, perm=[3,0,1,2])
    ys = np.array(ys)

    if one_hot:
        ys= np.array(ys).reshape(-1).astype(int)
        # 2 : seperate, contacted
        ys = np.eye(2)[ys]

# 256*256*3 2 pic --> 256*256*6 1 pic
# RODC_CNN.py
def RODC_model():
    with tf.name_scope('input2'):
        pre_X = tf.placeholder(tf.float32, [None, 256, 256, 6], name='pre_X')
        Y = tf.placeholder(tf.float32, [None, 2], name='Y')
        keep_prob = tf.placeholder(tf.float32)

    with tf.name_scope('conv1_max'):
        layer = batch_norm(X, False)
        layer = conv2d(layer, [3, 3, 6, 24], k_str=[1,2,2,1])
        layer = max_pool(layer)

    with tf.name_scope('conv2_max'):
        layer = batch_norm(layer)
        layer = conv2d(layer, [3, 3, 24, 72], k_str=[1,2,2,1])
        layer = max_pool(layer)

    with tf.name_scope('conv3_4'):
        layer = batch_norm(layer)
        layer = conv2d(layer, [3,3, 72, 125], k_str=[1,2,2,1])
        layer = conv2d(layer, [3,3, 125, 125], k_str=[1,2,2,1])

    with tf.name_scope('reshape'):
        reshaped = tf.reshape(layer, [-1, 4*4*125]) # 2000

    with tf.name_scope('FC'):
        layer = tf.layers.dense(reshaped, units=100 ,activation='Relu')
        layer = tf.nn.dropout(layer, keep_prob = keep_prob)
        model = tf.layers.dense(layer , units=2,activation='Relu')
        # layer = tf.nn.dropout(layer, keep_prob = keep_prob)

    with tf.name_scope('matrix'):
        x_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y))
        opt = tf.train.AdamOptimizer(LR).minimize(x_ent)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model,1),tf.argmax(Y,1)), tf.float32))

    return x_ent, opt, accuracy



    

    

