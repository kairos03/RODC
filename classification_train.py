# Copyright kairos03 2018. All Right Resrved.
"""
classification model
"""
import time
import tensorflow as tf

from data import process
from data import data_input

TOTAL_EPOCH = 100
BATCH_SIZE = 50
LEARNING_RATE = 1e-4
DROPOUT_RATE = 0.5
RANDOM_SEED = np.random.randint(0, 1000)

CURRENT = time.time()
LOG_TRAIN_PATH = 'log/' + str(CURRENT) + '/train/'
LOG_TEST_PATH = 'log/' + str(CURRENT) + '/test/'
MODEL_PATH = 'log/' + str(CURRENT) + '/model/'

pd = process.load_classification_dataset()
dataset = data_input.get_dataset(BATCH_SIZE) # TODO

def train():
    print('-----  training start  -----')

    with tf.Graph().as_default() as graph:

        with tf.name_scope('input'): 
            x = graph.get_tensor_by_name('inputs/x:0')  # input concat([left, right]) 
            is_combined = tf.placeholder(tf.float32, [None, 2])
            keep_prob = tf.placeholder(tf.float32)

        seg_out = graph.get_tensor_by_name('???:0') # TODO

        out_shape = seg_out.get_shape()
        front = seg_out[:out_shape[0]//2]
        right = seg_out[out_shape[0]//2:]

        # front
        conv1_f = tf.layers.conv2d(front, 64, [3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu)
        conv2_f = tf.layers.conv2d(conv1_f, 128, [3, 3], strides=(2, 2), padding='same', activation=tf.nn.relu)
        pool1_f = tf.layers.max_pooling2d(conv2_f, [2, 2], [2, 2], padding='same')

        conv3_f = tf.layers.conv2d(pool1_f, 128, [3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu)
        conv4_f = tf.layers.conv2d(conv3_f, 256, [3, 3], strides=(2, 2), padding='same', activation=tf.nn.relu)
        pool2_f = tf.layers.max_pooling2d(conv4_f, [2, 2], [2, 2], padding='same')

        # right
        conv1_r = tf.layers.conv2d(right, 64, [3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu)
        conv2_r = tf.layers.conv2d(conv1_r, 128, [3, 3], strides=(2, 2), padding='same', activation=tf.nn.relu)
        pool1_r = tf.layers.max_pooling2d(conv2_r, [2, 2], [2, 2], padding='same')

        conv3_r = tf.layers.conv2d(pool1_r, 128, [3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu)
        conv4_r = tf.layers.conv2d(conv3_r, 256, [3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu)
        pool2_r = tf.layers.max_pooling2d(conv4_r, [2, 2], [2, 2], padding='same')

        # concat
        concated = tf.concat([pool2_f, pool1_r], 3)

        conv5 = tf.layers.conv2d(concated, 512, [3, 3], strides=(1, 1), padding='same', activation=tf.nn.relu)
        conv6 = tf.layers.conv2d(conv5, 1024, [3, 3], strides=(2, 2), padding='same', activation=tf.nn.relu)
        pool3 = tf.layers.max_pooling2d(conv6, [2, 2], [2, 2], padding='same')

        resh = tf.reshape(pool3, (-1, 4 * 4 * 1024))

        dense1 = tf.layers.dense(resh, 1024, activation=tf.nn.relu)
        dense2 = tf.layers.dense(dense1, 256, activation=tf.nn.relu)
        drop = tf.layers.dropout(dense2, keep_prob)

        output = tf.layers.dense(drop, 2)

        with tf.name_scope('matrix'):
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=output, labels=is_combined), 'loss')
            optmizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
            acc = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(output, 1), tf.arg_max(is_combined, 1)), tf.float32), name='acc')

            tf.summary.scalar('loss', loss)
            tf.summary.scalar('accuracy', acc)

            saver = tf.train.Saver()
            merged = tf.summary.merge_all()

    with tf.Session(graph=graph) as sess:
        
        train_writer = tf.summary.FileWriter(LOG_TRAIN_PATH, graph=sess.graph)
        tf.global_variables_initializer().run()

        total_batch = dataset.total_batch

        for epoch in range(TOTAL_EPOCH):
            ep_loss = 0

            for _ in range(total_batch):
                
                #TODO
                x_s, _ = dataset.next_batch(RANDOM_SEED, valid_set=True)

                x_s, z_s = seg_pre_process(x_s)
                y_s = np.zeros((x_s.shape[0],3))

                _, summary, b_loss = sess.run(
                    [optimizer, merged, loss],
                    feed_dict={
                        x: x_s,
                        y: y_s,
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
