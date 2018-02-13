
import time

import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
import cv2

from data import data_input
from data import process
from data.process import seg_pre_process
from data.hb_process import cvtColor

TOTAL_EPOCH = 500
BATCH_SIZE = 15
LEARNING_RATE = 6e-5
DROPOUT_RATE = 0.8
RANDOM_SEED = np.random.randint(0, 1000)

CURRENT = time.time()
H_PRA = 'lr:{}_epoch:{}_batch:{}' .format(LEARNING_RATE, TOTAL_EPOCH, BATCH_SIZE)
LOG_TRAIN_PATH = 'log/' + str(CURRENT) + H_PRA + '/train/'
LOG_TEST_PATH = 'log/' + str(CURRENT) + H_PRA +'/test/'
MODEL_PATH = 'log/' + str(CURRENT) + H_PRA +'/model/'

df = process.load_fcn_train_dataset()
dataset = data_input.get_dataset(BATCH_SIZE, np.array(df['filename']),
                                 None, is_shuffle=True, is_valid=True)

n_classes = 3
input_shape = (256, 256)
label_shape = (256, 256)
deconv_weight = 2


def lrelu(x, leak=0.2, name="lrelu"):
    return tf.maximum(x, leak * x)


def fcn_loss(logits, labels, num_classes, head=None):
    """Calculate the loss from the logits and the labels.
    Args:
      logits: tensor, float - [batch_size, width, height, num_classes].
          Use vgg_fcn.upscore as logits.
      labels: Labels tensor, int32 - [batch_size, width, height, num_classes].
          The ground truth of your data.
      head: numpy array - [num_classes]
          Weighting the loss of each class
          Optional: Prioritize some classes
    Returns:
      loss: Loss tensor of type float.
    """
    with tf.name_scope('loss'):
        logits = tf.reshape(logits, (-1, num_classes))
        epsilon = tf.constant(value=1e-4)
        labels = tf.to_float(tf.reshape(labels, (-1, num_classes)))

        softmax = tf.nn.softmax(logits) + epsilon

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                                       head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        loss = tf.add_n(tf.get_collection('losses'), name='total_loss')
    return loss


def train():
    """
    model train
    """

    print('-----  training start  -----')

    with tf.Graph().as_default() as graph:

        # model
        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, 256, 256, 3], name='x')
            z = tf.placeholder(tf.float32, [None, 128, 128, 3], name='z')
            keep_prob = tf.placeholder(tf.float16)
            b_y = tf.truediv(z, 255.)

            tf.summary.image('input', x, 1)
            tf.summary.image('label', b_y, 1)

        with tf.variable_scope('convolution'):
            conv = tf.layers.batch_normalization(x)
            conv = tf.layers.conv2d(conv, 64, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            pool1 = tf.layers.max_pooling2d(
                conv, (2, 2), (2, 2), padding='same')

            conv = tf.layers.batch_normalization(pool1)
            conv = tf.layers.conv2d(conv, 128, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            pool2 = tf.layers.max_pooling2d(
                conv, (2, 2), (2, 2), padding='same')

            conv = tf.layers.batch_normalization(pool2)
            conv = tf.layers.conv2d(conv, 64, (1, 1), strides=(
                1, 1), padding='same', activation=lrelu)
            conv = tf.layers.conv2d(conv, 128, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            conv = tf.layers.conv2d(conv, 256, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            pool3 = tf.layers.max_pooling2d(
                conv, (2, 2), (2, 2), padding='same')

            conv = tf.layers.batch_normalization(pool3)
            conv = tf.layers.conv2d(conv, 128, (1, 1), strides=(
                1, 1), padding='same', activation=lrelu)
            conv = tf.layers.conv2d(conv, 256, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            conv = tf.layers.conv2d(conv, 512, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            pool4 = tf.layers.max_pooling2d(
                conv, (2, 2), (2, 2), padding='same')

            conv = tf.layers.batch_normalization(pool4)
            conv = tf.layers.conv2d(conv, 256, (1, 1), strides=(
                1, 1), padding='same', activation=lrelu)
            conv = tf.layers.conv2d(conv, 512, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            conv = tf.layers.conv2d(conv, 1024, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            pool5 = tf.layers.max_pooling2d(
                conv, (2, 2), (2, 2), padding='same')

        with tf.variable_scope('deconvolution'):
            up_sample4 = tf.layers.conv2d_transpose(
                pool5, 3, (5, 5), strides=(2, 2), padding='same', activation=lrelu)
            up_sample4 = tf.layers.conv2d_transpose(
                up_sample4, 3, (3, 3), strides=(1, 1), padding='same', activation=lrelu)
            up_pool4 = tf.layers.conv2d(pool4, 3, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            fuse1 = tf.add(up_sample4, up_pool4)

            up_sample3 = tf.layers.conv2d_transpose(
                fuse1, 3, (5, 5), strides=(2, 2), padding='same', activation=lrelu)
            up_sample3 = tf.layers.conv2d_transpose(
                up_sample3, 3, (3, 3), strides=(1, 1), padding='same', activation=lrelu)
            up_pool3 = tf.layers.conv2d(pool3, 3, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            fuse2 = tf.add(up_sample3, up_pool3)

            up_sample2 = tf.layers.conv2d_transpose(
                fuse2, 3, (5, 5), strides=(2, 2), padding='same', activation=lrelu)
            up_sample2 = tf.layers.conv2d_transpose(
                up_sample2, 3, (5, 5), strides=(1, 1), padding='same', activation=lrelu)
            up_pool2 = tf.layers.conv2d(pool2, 3, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            fuse3 = tf.add(up_sample2, up_pool2)

            up_sample1 = tf.layers.conv2d_transpose(
                fuse3, 3, (5, 5), strides=(2, 2), padding='same', activation=lrelu)
            up_sample1 = tf.layers.conv2d_transpose(
                up_sample1, 3, (5, 5), strides=(1, 1), padding='same', activation=lrelu)
            up_pool1 = tf.layers.conv2d(pool1, 3, (3, 3), strides=(
                1, 1), padding='same', activation=lrelu)
            fuse4 = tf.add(up_sample1, up_pool1)
            output = fuse4
            # pred = tf.argmax(output, dimension=3)

        # with tf.variable_scope('post_processing'):
        #     resh = tf.reshape(fuse3, (-1, 64*64, 3))
        #     preds = []
        #     for i in range(resh.shape[0]):
        #         pred = tf.zeros_like(i)
        #         pred[np.arange(len(i)), np.argmax(1)] = 1
        #         preds.append(pred)

        #     output=tf.stack(preds)

        with tf.name_scope('output'):
            tf.summary.image('output', output, 1)
            tf.summary.image('particle', tf.image.grayscale_to_rgb(tf.expand_dims(output[:, :, :, 0],-1)), 1)
            tf.summary.image('peptide', tf.image.grayscale_to_rgb(tf.expand_dims(output[:, :, :, 1],-1)), 1)
            tf.summary.image('background', tf.image.grayscale_to_rgb(tf.expand_dims(output[:, :, :, 2],-1)), 1)
            # tf.summary.image('pred', pred, 1)

        print('input', x.shape)
        print('z', z.shape)
        print('pool2', pool2.shape)
        print('pool3', pool3.shape)
        print('pool4', pool4.shape)
        print('pool5', pool5.shape)
        print('fuse1', fuse1.shape)
        print('fuse2', fuse2.shape)
        print('fues3', fuse3.shape)
        print('output', output.shape)

        with tf.name_scope('matrix'):

            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            #     logits=output, labels=b_y))
            loss = fcn_loss(output, b_y, 3)

            optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

            # re_out = tf.reshape(output, (-1, 3))
            # re_y = tf.reshape(b_y, (-1, 3))

            # print(tf.argmax(re_out, 1).get_shape())
            # print(tf.argmax(re_y, 1).get_shape())

            # accuracy = tf.cast(tf.equal(tf.argmax(output, 3), tf.argmax(b_y, 3)), tf.float32)

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('learning_rate', LEARNING_RATE)
            # tf.summary.scalar('accuracy', accuracy)

            saver = tf.train.Saver()
            merged = tf.summary.merge_all()

    with tf.Session(graph=graph) as sess:

        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, graph=sess.graph)
        tf.global_variables_initializer().run()

        total_batch = dataset.total_batch

        for epoch in range(TOTAL_EPOCH):
            ep_loss = 0

            for _ in range(total_batch):

                x_s, _ = dataset.next_batch(RANDOM_SEED, valid_set=True)

                x_s, z_s = seg_pre_process(x_s)

                _, summary, b_loss = sess.run(
                    [optimizer, merged, loss],
                    feed_dict={
                        x: x_s,
                        # y: y_s,
                        z: z_s,
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
    x_s, y_s = pre_process(x_s)

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
                                             z: y_s,
                                             keep_porb: 1.0})
            total_loss += xent

        print("TEST LOSS: %.5f" % (total_loss / dataset.valid_total_batch))
        print('-----  test end  -----')


if __name__ == '__main__':

    train()
    test()
