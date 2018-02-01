# Copyright 2018 Kairos03. All Right Reserved.
"""
train
"""

import time

import tensorflow as tf
import numpy as np

from data import data_input
from data import process

import fcn_model

TOTAL_EPOCH = 1000
BATCH_SIZE = 50
LEARNING_RATE = 1e-6
DROPOUT_RATE = 0.9
RANDOM_SEED = np.random.randint(0, 1000)

CURRENT = time.time()
LOG_TRAIN_PATH = 'log/' + str(CURRENT) + '/train/'
LOG_TEST_PATH = 'log/' + str(CURRENT) + '/test/'
MODEL_PATH = 'log/' + str(CURRENT) + '/model/'

df = process.load_fcn_train_dataset()
dataset = data_input.get_dataset(BATCH_SIZE, np.array(df['filename']),
                                 None, is_shuffle=True, is_valid=True)


def train():
    """
    model train
    """

    print('-----  training start  -----')

    with tf.Graph().as_default() as train_g:
        # model
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='x')
            y = tf.placeholder(tf.float32, [None, 256, 256, 3], name='y')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        _, loss, optimizer = fcn_model.make_model(
            x, y, keep_prob, LEARNING_RATE)

        with tf.variable_scope('matrix'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('learning_rate', LEARNING_RATE)

            saver = tf.train.Saver()
            merged = tf.summary.merge_all()

    with tf.Session(graph=train_g) as sess:

        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, graph=sess.graph)
        tf.global_variables_initializer().run()

        total_batch = dataset.total_batch

        for epoch in range(TOTAL_EPOCH):
            ep_loss = 0

            for _ in range(total_batch):

                x_s, _ = dataset.next_batch(RANDOM_SEED, valid_set=True)

                x_s, y_s = fcn_model.pre_process(x_s)

                _, summary, b_loss = sess.run(
                    [optimizer, merged, loss],
                    feed_dict={
                        x: x_s,
                        y: y_s,
                        keep_prob: DROPOUT_RATE
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


def test():
    """
    model test
    """

    print('-----  test start  -----')

    x_s, y_s = dataset.next_batch(RANDOM_SEED, valid_set=True)
    x_s, y_s = fcn_model.pre_process(x_s)

    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(
        MODEL_PATH + '-' + str(TOTAL_EPOCH) + '.meta')

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL_PATH))
        graph = tf.get_default_graph()

        # place holder
        x = graph.get_tensor_by_name('input/x:0')
        y = graph.get_tensor_by_name('input/y:0')
        keep_porb = graph.get_tensor_by_name('input/keep_prob:0')

        loss = graph.get_tensor_by_name('matrix/loss:0')
        # loss = sess.run('matrix/loss:0')

        total_loss = 0

        for batch in range(dataset.valid_total_batch):

            xent = sess.run(loss, feed_dict={x: x_s,
                                             y: y_s,
                                             keep_porb: 1.0})
            total_loss += xent

        print("TEST LOSS: %.5f" % (total_loss / dataset.valid_total_batch))
        print('-----  test end  -----')


if __name__ == '__main__':

    train()
    test()
