
import time

import tensorflow as tf
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from tensorflow.python.util import freeze

from data import data_input
from data import process
from data.process import seg_pre_process


<<<<<<< HEAD

TOTAL_EPOCH = 10000
=======
TOTAL_EPOCH = 3000
>>>>>>> ceb92f16a0637cf37e0f7697b09bb332062f5303
BATCH_SIZE = 50
LEARNING_RATE = 1e-3
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
    # feature = None

    # if feature == None:
    with tf.Graph().as_default() as train_g2:
        #with tf.name_scope('re_feature'):

        # conv 
        saver = tf.train.import_meta_graph('/home/kairos03/RODC/log/feature_model2/model/-100.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('/home/kairos03/RODC/log/feature_model2/model/'))

        graph = tf.get_default_graph()
        pool1 = graph.get_tensor_by_name('pool1/MaxPool:0')
        pool2 = graph.get_tensor_by_name('pool2/MaxPool:0')
        pool3 = graph.get_tensor_by_name('pool3/MaxPool:0')
        # print(pool3)

<<<<<<< HEAD
        # model
        with tf.name_scope('input'):
            x = graph.get_tensor_by_name('input/x:0')
            y = tf.placeholder(tf.float32, [None, 256, 256, 3], name='y')
            keep_prob = graph.get_tensor_by_name('input/keep_prob:0')

            tf.summary.image('input', x, 1)
            tf.summary.image('label', y, 1)

            zero = np.zeros((1, 256, 256, 1))
            '''r = tf.concat([tf.reshape(y[0,:,:,0], (1,256,256,1)), zero, zero], axis=3)
            g = tf.concat([zero, tf.reshape(y[0,:,:,1], (1,256,256,1)), zero], axis=3)
            b = tf.concat([zero, zero, tf.reshape(y[0,:,:,2], (1,256,256,1))], axis=3)

            tf.summary.image('r', r, 1)
            tf.summary.image('g', g, 1)
            tf.summary.image('b', b, 1)'''

            b_y = tf.truediv(y, 255.)
            

        with tf.variable_scope('fcn'):
            conv5 = tf.layers.conv2d(pool3, 2048, kernel_size=[1, 1], padding='same', activation=lrelu)
=======
    frozen_graph = 'log/1517567622.9337113/model/-100.pb'
    with tf.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as feature_graph:
        tf.import_graph_def(graph_def,
                            input_map=None,
                            return_elements=None,
                            name=""
                            )

        # model
        with tf.name_scope('fcn_input'):
            x = feature_graph.get_tensor_by_name('input/x:0')
            # y = feature_graph.get_tensor_by_name('input/y:0')
            z = tf.placeholder(tf.float32, [None, 64, 64, 3], name='z')
            keep_prob = tf.placeholder(tf.float16)
            b_y = tf.truediv(z, 255.)
            # b_y = tf.arg_max(z)

            tf.summary.image('input', x, 1)
            tf.summary.image('label', b_y, 1)

            

            pool2 = feature_graph.get_tensor_by_name('pool2/MaxPool:0')
            pool3 = feature_graph.get_tensor_by_name('pool3/MaxPool:0')
            feature_output = feature_graph.get_tensor_by_name('pool4/MaxPool:0')
            # pool2 = tf.stop_gradient(pool2)
            # pool3 = tf.stop_gradient(pool3)
            # feature_output = tf.stop_gradient(feature_output)

        with tf.variable_scope('fcn'):
            conv5 = tf.layers.conv2d(feature_output, 2048, kernel_size=[
                                    1, 1], padding='same', activation=lrelu)
>>>>>>> ceb92f16a0637cf37e0f7697b09bb332062f5303
            drop1 = tf.layers.dropout(conv5, keep_prob)
            conv6 = tf.layers.conv2d(drop1, 2048, kernel_size=[
                                    1, 1], padding='same', activation=lrelu)
            drop2 = tf.layers.dropout(conv6, keep_prob)
            conv7 = tf.layers.conv2d(drop2, 1024, kernel_size=[
                                    1, 1], padding='same', activation=lrelu)  # TODO

        # deconv
<<<<<<< HEAD
        bn = tf.layers.batch_normalization(conv7)
        deconv1 = tf.layers.conv2d_transpose(bn, 386, kernel_size=[4, 4], strides=[2, 2], padding='same', activation=lrelu)
        reshape = tf.reshape(pool3, [-1, 16,16,386])
        fuse1 = tf.add(deconv1, reshape)

        bn = tf.layers.batch_normalization(fuse1)
        deconv2 = tf.layers.conv2d_transpose(bn, 128, kernel_size=[4, 4], strides=[2, 2], padding='same', activation=lrelu)
        reshape = tf.reshape(pool2, [-1, 32,32,128])
<<<<<<< Updated upstream
        fuse2 = tf.add(deconv2, reshape)
=======
        fuse2 = tf.add(deconv2, pool2)
>>>>>>> Stashed changes

        bn = tf.layers.batch_normalization(fuse2)
        output = tf.layers.conv2d_transpose(bn, 3, kernel_size=[8, 8], strides=[4, 4], padding='same', activation=lrelu)


        with tf.name_scope('output'):
            '''zero = np.zeros((1, 256, 256, 1))
            r = tf.concat([tf.reshape(output[0,:,:,0], (1,256,256,1)), zero, zero], axis=3)
            g = tf.concat([zero, tf.reshape(output[0,:,:,1], (1,256,256,1)), zero], axis=3)
            b = tf.concat([zero, zero, tf.reshape(output[0,:,:,2], (1,256,256,1))], axis=3)

            tf.summary.image('output', output, 1)
            tf.summary.image('r', r, 1)
            tf.summary.image('g', g, 1)
            tf.summary.image('b', b, 1)'''
=======
        # conv7 = tf.nn.tanh(conv7)
        deconv1 = tf.layers.conv2d_transpose(conv7, 512, kernel_size=[3, 3], strides=[
                                            2, 2], padding='same')
        # bn = tf.layers.batch_normalization(deconv1)
        act = lrelu(deconv1)
        fuse1 = tf.add(act, pool3)

        deconv2 = tf.layers.conv2d_transpose(fuse1, 256, kernel_size=[3, 3], strides=[
                                            2, 2], padding='same')
        # bn = tf.layers.batch_normalization(deconv2)
        act = lrelu(deconv2)
        fuse2 = tf.add(act, pool2)

        deconv3 = tf.layers.conv2d_transpose(fuse2, 3, kernel_size=[3, 3], strides=[
                                            2, 2], padding='same', name='')
        # bn = tf.layers.batch_normalization(deconv3)
        act = lrelu(deconv3)

        output = tf.layers.conv2d_transpose(act, 3, kernel_size=[3, 3], strides=[
                                            2, 2], padding='same', name='')
        # output = tf.layers.batch_normalization(output)
        output = lrelu(output)


        with tf.name_scope('output'):
            # zero = np.zeros((1, 256, 256, 1))
            # r = tf.concat([tf.reshape(output[0,:,:,0], (1,256,256,1)), zero, zero], axis=3)
            # g = tf.concat([zero, tf.reshape(output[0,:,:,1], (1,256,256,1)), zero], axis=3)
            # b = tf.concat([zero, zero, tf.reshape(output[0,:,:,2], (1,256,256,1))], axis=3)

            tf.summary.image('output', output, 1)
            # tf.summary.image('r', r, 1)
            # tf.summary.image('g', g, 1)
            # tf.summary.image('b', b, 1)
>>>>>>> ceb92f16a0637cf37e0f7697b09bb332062f5303

        print('input', x.shape)
        print('z', z.shape)
        print('pool2', pool2.shape)
        print('pool3', pool3.shape)
        print('pool4', feature_output.shape)
        print('conv7', conv7.shape)
        print('fuse1', fuse1.shape)
        print('fuse2', fuse2.shape)
        print('output', output.shape)

        with tf.name_scope('matrix'):
<<<<<<< HEAD
            
            # r_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output[:,:,:,0], labels=tf.stop_gradient(b_y[:,:,:,0]))
            # g_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output[:,:,:,1], labels=tf.stop_gradient(b_y[:,:,:,1]))
            # b_loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=output[:,:,:,2], labels=tf.stop_gradient(b_y[:,:,:,2]))

            # loss = 5*r_loss + 5*g_loss + 1*b_loss
            # loss = .5 * tf.reduce_mean(loss , name='loss')
            # loss = tf.reduce_mean(g_loss)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=output, labels=tf.stop_gradient(y)), name='loss')
=======

            # loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            #     logits=output, labels=b_y))
            loss = fcn_loss(output, b_y, 3)

>>>>>>> ceb92f16a0637cf37e0f7697b09bb332062f5303
            optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('learning_rate', LEARNING_RATE)

            saver = tf.train.Saver()
            merged = tf.summary.merge_all()
            
<<<<<<< Updated upstream
            
=======
>>>>>>> Stashed changes
            #sess.close()

<<<<<<< HEAD
    with tf.Session(graph=train_g2) as sess:
=======
    with tf.Session(graph=feature_graph) as sess:
>>>>>>> ceb92f16a0637cf37e0f7697b09bb332062f5303

        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, graph=sess.graph)
        tf.global_variables_initializer().run()

        tf.variables_initializer([x, ]).run()

        total_batch = dataset.total_batch

        for epoch in range(TOTAL_EPOCH):
            ep_loss = 0

            for _ in range(total_batch):

                x_s, _ = dataset.next_batch(RANDOM_SEED, valid_set=True)

                x_s, z_s = seg_pre_process(x_s)
                y_s = np.zeros((x_s.shape[0], 3))

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
        keep_prob = graph.get_tensor_by_name('input/keep_prob:0')

        loss = graph.get_tensor_by_name('matrix/loss:0')
        # loss = sess.run('matrix/loss:0')

        total_loss = 0

        for batch in range(dataset.valid_total_batch):

            xent = sess.run(loss, feed_dict={x: x_s,
<<<<<<< HEAD
                                             y: y_s,
                                             keep_prob: 1.0})
=======
                                             z: y_s,
                                             keep_porb: 1.0})
>>>>>>> ceb92f16a0637cf37e0f7697b09bb332062f5303
            total_loss += xent

        print("TEST LOSS: %.5f" % (total_loss / dataset.valid_total_batch))
        print('-----  test end  -----')


if __name__ == '__main__':

    train()
    test()
