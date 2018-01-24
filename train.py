# Copyright 2018 Kairos03. All Right Reserved.
"""
train
"""

import time

import tensorflow as tf
import numpy as np

from data import data_input
from data import process

import model

TOTAL_EPOCH = 100
BATCH_SIZE = 50
LEARNING_RATE = 1e-7
DROPOUT_RATE = 0.5
RANDOM_SEED = np.random.randint(0, 1000)

CURRENT = time.time()
LOG_TRAIN_PATH = 'log/' + str(CURRENT) + '/train/'
LOG_TEST_PATH = 'log/' + str(CURRENT) + '/test/'
MODEL_PATH = 'log/' + str(CURRENT) + '/model/'

images, labels = process.load_image_train_dataset()
dataset = data_input.get_dataset(
    BATCH_SIZE, images, labels, is_shuffle=True, is_valid=True)


def train():
    """
    model train
    """

    print('-----  training start  -----')

    with tf.Graph().as_default() as train_g:
        # model
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 320, 320, 3], name='x')
            y = tf.placeholder(tf.float32, [None, 2], name='y')
            keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        _, loss, optimizer, accuracy = model.make_model(
            x, y, keep_prob, LEARNING_RATE)

        with tf.variable_scope('matrix'):
            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', accuracy)
            tf.summary.scalar('learning_rate', LEARNING_RATE)

            saver = tf.train.Saver()
            merged = tf.summary.merge_all()

    with tf.Session(graph=train_g) as sess:

        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, graph=sess.graph)
        tf.global_variables_initializer().run()

        total_batch = dataset.total_batch

        for epoch in range(TOTAL_EPOCH):
            ep_loss, ep_acc = 0, 0

            for _ in range(total_batch):

                x_s, y_s = dataset.next_batch(RANDOM_SEED, valid_set=True)

                _, summary, b_loss, b_acc = sess.run(
                    [optimizer, merged, loss, accuracy],
                    feed_dict={
                        x: x_s,
                        y: y_s,
                        keep_prob: DROPOUT_RATE
                    })

                ep_loss += b_loss
                ep_acc += b_acc

            train_writer.add_summary(summary, global_step=epoch)

            if epoch == 0 or epoch % 20 == 19:
                print("[EP %3d] loss: %.5f acc: %.5f" %
                      (epoch, ep_loss / total_batch, ep_acc / total_batch))
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

        accuracy = graph.get_tensor_by_name('matrix/accuracy:0')
        # loss = sess.run('matrix/loss:0')

        total_acc = 0

        for batch in range(dataset.valid_total_batch):

            acc = sess.run(accuracy, feed_dict={x: x_s,
                                                y: y_s,
                                                keep_porb: 1.0})
            total_acc += acc

        print("TEST ACCURACY: %.5f" % (total_acc / dataset.valid_total_batch))
        print('-----  test end  -----')


if __name__ == '__main__':
    train()
    test()
