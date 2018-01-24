#

import time

import tensorflow as tf
import numpy as np

from data import process
from data import data_input
import feature_model

# Hyper-parameters
TOTAL_EPOCH = 500
BATCH_SIZE = 100
LEARNING_RATE = 1e-3
DROPOUT_RATE = 0.9
RANDOM_SEED = int(np.random.random() * 1000)

CURRENT = time.time()
LOG_TRAIN_PATH = 'log/'+str(CURRENT)+'/train/'
LOG_TEST_PATH = 'log/'+str(CURRENT)+'/test/'
MODEL_PATH = 'log/'+str(CURRENT)+'/model/'


def train(is_valid=True):
    # data set load
    x, y = process.load_image_train_dataset()

    # make dataset
    inputs = data_input.get_dataset(
        BATCH_SIZE, x, y, is_shuffle=True, is_valid=is_valid)

    # input placeholder
    with tf.name_scope('input'):
        X = tf.placeholder(tf.float32, [None, 320, 320, 3], name='X')
        Y = tf.placeholder(tf.float32, [None, 2], name='Y')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # get train ops
    _, xent, optimizer, accuracy = feature_model.make_model(X, Y, keep_prob, LEARNING_RATE)

    print()
    print('Hyper Params')
    print("====================================================")
    print('Batch Size', BATCH_SIZE)
    print('Dropout Rate', DROPOUT_RATE)
    print('Random Seed', RANDOM_SEED)
    print('\n')

    # train start
    print('Train Start')

    # define saver and summary marge
    saver = tf.train.Saver()
    merged = tf.summary.merge_all()

    # start session
    with tf.Session() as sess:

        # initalize global variables
        init = tf.global_variables_initializer()
        sess.run(init)

        # define summary writer and write graph
        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, sess.graph)
        test_writer = tf.summary.FileWriter(LOG_TEST_PATH)

        # total_batch
        total_batch = inputs.total_batch
        if is_valid:
            valid_total_batch = inputs.valid_total_batch

        # get probabiliy op
        # probability = sess.graph.get_tensor_by_name('matrices/proba:0')

        for epoch in range(TOTAL_EPOCH):
            
            # initialize epoch variable
            epoch_loss = epoch_acc = 0
            xs = ys = None

            # start batch
            for batch_num in range(total_batch):
                
                # get data
                xs, ys = inputs.next_batch(RANDOM_SEED, valid_set=False)

                # run ops
                _, loss, acc = sess.run([optimizer, xent, accuracy],
                                        feed_dict={X: xs, Y: ys, keep_prob: DROPOUT_RATE})
                
                # sum batch loss and accuracy
                epoch_loss += loss
                epoch_acc += acc

            # write train summary
            summary = sess.run(merged,
                               feed_dict={X: xs, Y: ys, keep_prob: DROPOUT_RATE})
            train_writer.add_summary(summary, epoch)

            # calculate epoch loss, acurracy and display its
            epoch_loss = epoch_loss / total_batch
            epoch_acc = epoch_acc / total_batch
            if epoch % 20 == 19 or epoch == 0:
                print('[{:05.3f}] TRAIN EP: {:05d} | loss: {:0.5f} | acc: {:0.5f}'
                      .format(time.time() - CURRENT, epoch, epoch_loss, epoch_acc))

        # model save
        saver.save(sess, MODEL_PATH)

    print('Train Finish')

if __name__ == '__main__':
    train()