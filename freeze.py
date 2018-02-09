<<<<<<< HEAD
import numpy as np 
import tensorflow as tf 
from tensorflow.python.platform import gfile
from tensorflow.python.framework.graph_util import convert_variables_to_constants

def save(sess):
    saver = tf.train.Saver()
    saver.save(sess, '', write_meta_graph=True)

    with open('./tm'+'d.pb','wb') as f:
        t.write(sess.graph_def.SerializeToString())
    sess.close()

def freeze():
    saver = tf.train.import_meta_graph('/home/kairos03/RODC/log/feature_model2/model/-100.meta')
    with tf.Session() as sess:
        saver.restore(sess,'/home/kairos03/RODC/log/feature_model2/model/-100.meta')

        # customize net
    with gfile.FastGFile('/home/kairos03/RODC/log/'+'frozen.pb', 'rb') as f:
        graph_def = tf.GraphDef()   # graph container create
        graph_def.PaserFromString(f.read()) # read
    frozen_graph_def = convert_variables_to_constants(sess, )



    # if feature == None:
    with tf.Graph().as_default() as train_g2:
        #with tf.name_scope('re_feature'):

        # conv 
       
        # saver.restore(sess, tf.train.latest_checkpoint('/home/kairos03/RODC/log/feature_model2/model/'))

        graph = tf.get_default_graph()
        pool1 = graph.get_tensor_by_name('pool1/MaxPool:0')
        pool2 = graph.get_tensor_by_name('pool2/MaxPool:0')
        pool3 = graph.get_tensor_by_name('pool3/MaxPool:0')
        # print(pool3)

        # model
        with tf.name_scope('input'):
            x = graph.get_tensor_by_name('input/x:0')
            y = tf.placeholder(tf.float32, [None, 256, 256, 3], name='y')
            keep_prob = graph.get_tensor_by_name('input/keep_prob:0')

            tf.summary.image('input', x, 1)
            tf.summary.image('label', y, 1)
=======
"""
Copyright 2018 kairos03. All Right Reserved.
"""

import tensorflow as tf

feature_path = "log/1517567622.9337113/model/"
feature_outputs = "pool2/MaxPool,pool3/MaxPool,pool4/MaxPool"

def freeze_model(model_path, output_node_names):
    saver = tf.train.import_meta_graph(model_path+'-100.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    sess = tf.Session()
    last_ckpt = tf.train.latest_checkpoint(model_path)
    saver.restore(sess, last_ckpt)

    # [print(n.name) for n in tf.get_default_graph().as_graph_def().node]
    
    output_graph_def = tf.graph_util.convert_variables_to_constants(
                sess, # The session
                input_graph_def, # input_graph_def is useful for retrieving the nodes 
                output_node_names.split(",")  
    )

    output_graph=last_ckpt+".pb"
    with tf.gfile.GFile(output_graph, "wb") as f:
        f.write(output_graph_def.SerializeToString())
    
    sess.close()


if __name__ == '__main__':
    freeze_model(feature_path, feature_outputs)
>>>>>>> ceb92f16a0637cf37e0f7697b09bb332062f5303
