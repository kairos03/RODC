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
