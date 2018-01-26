# Copyright 2018 kairos03. All Right Reserved.

import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np


# constant 
S = 7       # grid width and height
SS = S*S    # number of cell
B = 2       # predicted Box number
C = 2       # classes of the objects

# loss const
sporb = 1
sconf = 1
snoob = .5
scoor = 5

def leaky_relu(x, alpha=0.1):
    """
    leaky relu
    """
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


def loss(logits, labels):
    """
    yolo loss
    """

    # true labels
    _probs = labels['probs']
    _confs = labels['confs']
    _coord = labels['coord']

    _proid = labels['proid']
    _areas = labels['areas']
    _upleft = labels['upleft']
    _botright = labels['botright']

    # Extract coordinate prediction from logits
    coords = logits[:, SS * (B + C):] # coords (x,y,w,h)
    coords = tf.reshape(coords, (-1, SS, B, 4)) 
    wh = tf.pow(coords[:,:,:,2:4], 2) * S # size of width and height of each Boxes 
    area_pred = wh[:,:,:,0] * wh[:,:,:,1] # areas of each Boxes
    center = coords[:,:,:,0:2]  # x, y
    upleft = center - (wh * .5) # boxes upleft 
    botright = center + (wh * .5) # boxes botright
    # Note. multiplication is fester then division, in python

    # calculate intersection areas
    intersect_upleft = tf.maximum(upleft, _upleft) 
    intersect_botright = tf.minimum(botright, _botright)
    intersect_wh = intersect_botright - intersect_upleft
    intersect_wh = tf.maximum(intersect_wh, 0.0)
    intersect = tf.multiply(intersect_wh[:,:,:,0], intersect_wh[:,:,:,1]) # areas of intersect

    # calculate IOU, set 0.0 confidence for worse boxes
    iou = tf.truediv(intersect, _areas, + area_pred - intersect)
    best_box = tf.equal(iou, tf.reduce_max(iou, [2], True))
    tf.to_float(best_box)
    confs = tf.multiply(best_box, _confs)

    # weights
    conid = sconf * confs + snoob * (1. - confs)  # 1 * 1obj(i,j) + lambda(noobj) * (1 - 1obj(i,j)), 1 - 1obj(i,j) = 1noobj(i,j) 
    weight_coo = tf.concat(4 * [tf.expand_dims(confs, -1)], 3) # [-1, SS, B, 4] of 1obj(i,j)
    cooid = scoor * weight_coo
    proid = sporb * _proid

    # flatten
    probs = slim.flatten(_probs)
    proid = slim.flatten(proid)
    confs = slim.flatten(confs)
    conid = slim.flatten(conid)
    coord = slim.flatten(_coord)
    cooid = slim.flatten(cooid)

    # calculate loss
    grand_truth = tf.concat([probs, confs, coord], 1)
    weight = tf.concat([proid, conid, cooid], 1)
    loss = tf.pow(logits - grand_truth, 2)
    loss = tf.multiply(loss, weight)
    loss = tf.reduce_sum(loss, 1)
    loss = .5 * tf.reduce_mean(loss)

    return loss
    

def make_model(inputs, hparam):
    """
    detection model
    """

    # inputs
    images = inputs['images']
    #...

    # hparam
    # ...

    # feature model load

    feature_out = np.zeros((1, 10, 10, 1024))  # TODO fake output

    # add layer
    conv1 = tf.layers.conv2d(feature_out, 1024, (3, 3),
                             strides=(1, 1),
                             padding='same',
                             activation=leaky_relu)
    conv2 = tf.layers.conv2d(conv1, 1024, (3, 3),
                             strides=(1, 1),
                             padding='same',
                             activation=leaky_relu)

    reshape = tf.reshape(conv2, (-1, 10 * 10 * 1024))

    conn1 = tf.layers.dense(reshape, 4096,
                            activation=leaky_relu)

    # output = (S * S * ((box_param * box_num) + classes)
    output = tf.layers.dense(conn1, (-1, SS * ((5 * B) + C)), 
                             activation=leaky_relu)


