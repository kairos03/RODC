# copyright 2018 quisutdeus7 all reserved
# implement : https://arxiv.org/pdf/1506.02640.pdf
import tensorflow as tf
import os
import sys
import numpy as np
import tf_fcn_8 as fcn8
import loss
import time

from data import data_input
from data import process
from data.process import seg_pre_process

# model ingredients
# parameter
TOTAL_EPOCH = 10000
BATCH_SIZE = 50
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.9
RANDOM_SEED = np.random.randint(0, 1000)

CURRENT = time.time()
LOG_TRAIN_PATH = 'log/' + str(CURRENT) + '/train/'
LOG_TEST_PATH = 'log/' + str(CURRENT) + '/test/'
MODEL_PATH = 'log/' + str(CURRENT) + '/model/'

df = process.load_fcn_train_dataset()
dataset = data_input.get_dataset(BATCH_SIZE, np.array(df['filename']),
                                 None, is_shuffle=True, is_valid=True)

n_classes = 3
input_shape = (256, 256)
label_shape = (256, 256)
deconv_weight = 2

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
# 256*256*3 2 pic --> 256*256*6 1 pic
# RODC_CNN.py
def RODC_model(pre_x, Y, keep_prob):
    '''
    with tf.name_scope('input_2'):
        pre_X = tf.placeholder(tf.float32, [None, 256, 256, 6], name='pre_X')
        Y = tf.placeholder(tf.float32, [None, 2], name='Y')
        keep_prob = tf.placeholder(tf.float32)
    '''
    with tf.name_scope('conv1_max'):
        layer = batch_norm(pre_x, False)
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
        x_ent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=model, labels=Y), name='x_ent')
        opt = tf.train.AdamOptimizer(LR, name="opt").minimize(x_ent)
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(model,1),tf.argmax(Y,1)), tf.float32), name='accuracy')

    return output, x_ent, opt, accuracy

def train():
    
    vgg8 = fcn8.FCN8VGG(vgg16_npy_path='/home/mike2ox/RODC/data/vgg16.npy')
    x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='x')
    y = tf.placeholder(tf.float32, [None, 256, 256, 3], name='y')

    vgg8.build(x, True, num_classes=3)
    vgg_loss = loss.loss(logits=vgg8.pred_up, labels=y, num_classes=3)
    print(vgg_loss)
    ## optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(vgg_loss)

    tf.summary.scalar('loss', vgg_loss)
    tf.summary.scalar('learning_rate', LEARNING_RATE)

    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, graph=sess.graph)
        tf.global_variables_initializer().run()

        # tf.variables_initializer([x, ]).run()

        total_batch = dataset.total_batch

        for epoch in range(TOTAL_EPOCH):
            ep_loss = 0

            for _ in range(total_batch):

                x_s, _ = dataset.next_batch(RANDOM_SEED, valid_set=True)

                x_s, y_s = seg_pre_process(x_s)

                summary, b_loss = sess.run(
                    [merged, vgg_loss],
                    feed_dict={
                        x: x_s,
                        y: y_s,
                    })

                ep_loss += b_loss

            train_writer.add_summary(summary, global_step=epoch)

            if epoch == 0 or epoch % 10 == 9:
                print("[EP %3d] loss: %.5f" %
                      (epoch, ep_loss / total_batch))
                saver.save(sess, MODEL_PATH, global_step=epoch)

        # final save
        saver.save(sess, MODEL_PATH, global_step=TOTAL_EPOCH)

    print('-----  training end  -----')

if __name__ == '__main__':

    train()
