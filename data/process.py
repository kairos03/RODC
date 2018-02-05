# Copyright 2018 kairos03. All Right Reserved.

"""
Image data process (labeling)
"""
import os
import pickle

import xml.etree.ElementTree as ET

from scipy import misc
import numpy as np
import pandas as pd
import h5py


DATA_PATH = 'data/seg_data/'

PARTICLE = [1.,0.,0.]
PEPTIDE = [0.,1.,0.]
BACKGROUND = [0.,0.,1.]

input_shape = (256, 256)
label_shape = (256, 256)

PEPTIDE_PATH = DATA_PATH + 'peptide/'
PARTICLE_PATH = DATA_PATH + 'particle/'
BACKGROUND_PATH = DATA_PATH + 'background/'
IMAGE_TRAIN_DATASET_PATH = DATA_PATH + 'image_train.h5'

DETACTION_OBJECTS_PATH = DATA_PATH + 'objects/'
DETACTION_TRAIN_DATASET_PATH = DATA_PATH + 'detation_train.h5'

ORIGIN_PATH = DATA_PATH + 'image/'
MASK_PATH = DATA_PATH + 'mask/'
ANNO_PATH = DATA_PATH + 'anno/'
FCN_TRAIN_DATASET_PATH = DATA_PATH + 'fcn_train.h5'


def make_image_train_dataset(path=IMAGE_TRAIN_DATASET_PATH):
    """
    """
    df = pd.DataFrame()
    pep = os.listdir(PEPTIDE_PATH)
    par = os.listdir(PARTICLE_PATH)
    bag = os.listdir(BACKGROUND_PATH)

    # label to anno_img
    for name in par:
        df = df.append({'filename': PARTICLE_PATH + name, 'class': PARTICLE},
                       ignore_index=True)

    for name in pep:
        df = df.append({'filename': PEPTIDE_PATH + name, 'class': PEPTIDE},
                       ignore_index=True)

    for name in bag:
        df = df.append({'filename': BACKGROUND_PATH + name, 'class': BACKGROUND},
                       ignore_index=True)

    # save
    df.to_hdf(path, 'feature')
    print("SAVED", path, df.shape)


def load_image_train_dataset(path=IMAGE_TRAIN_DATASET_PATH):
    """
    """
    df = pd.read_hdf(path)
    print("LOADED", path, df.shape)
    return df


def make_detacion_train_dataset(path=DETACTION_OBJECTS_PATH):
    """
    """
    annotaion_path = 'Annotation/'

    dataset = pd.DataFrame()

    for xml_path in os.listdir(DETACTION_OBJECTS_PATH + annotaion_path):

        # Annotaion XML parsing
        tree = ET.parse(DETACTION_OBJECTS_PATH + annotaion_path + xml_path)
        root = tree.getroot()

        # extract img_filename and coord
        img_filename = root.find('filename').text
        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        for obj in root.findall('object'):
            name = obj.find('name').text
            bndbox = obj.find('bndbox')

            xmin = int(bndbox.find('xmin').text)
            ymin = int(bndbox.find('ymin').text)
            xmax = int(bndbox.find('xmax').text)
            ymax = int(bndbox.find('ymax').text)

            # append data
            dataset = dataset.append(
                {'filename': img_filename, 'width': width, 'height': height, 'name': name,
                 'xmin': xmin, 'xmax': xmax, 'ymin': ymin, 'ymax': ymax}, ignore_index=True)

    # save
    dataset.to_hdf(DETACTION_TRAIN_DATASET_PATH, 'detacion')
    # print(dataset.head)
    print("SAVED", DETACTION_TRAIN_DATASET_PATH, dataset.shape)


def load_detacion_train_dataset(path=DETACTION_TRAIN_DATASET_PATH):
    """
    """
    df = pd.read_hdf(path)
    print("LOADED", path, df.shape)
    return df


def image_to_anno(img, width=256, height=256):

    mask = (img == np.reshape(np.amax(img, 2), (width, height, 1))).astype(float)

    for i in range(width):
        for j in range(height):
            if (mask[i, j] == [1., 1., 1.]).all():
                mask[i, j] = [0., 0., 1.]

    return mask


def make_fcn_train_dataset(path=FCN_TRAIN_DATASET_PATH):
    df = pd.DataFrame()
    filenames = np.array(os.listdir(ORIGIN_PATH))
    df['filename'] = filenames

    # label to anno_img
    for name in filenames:
        img = misc.imread(MASK_PATH + name)
        img = misc.imresize(img, (256, 256))
        img = image_to_anno(img)
        misc.imsave(ANNO_PATH + name, img)

    # save
    df.to_hdf(path, 'detacion')
    print("SAVED", path, df.shape)


def load_fcn_train_dataset(path=FCN_TRAIN_DATASET_PATH):
    """
    """
    df = pd.read_hdf(path)
    print("LOADED", path, df.shape)
    return df


def pre_process(image_names, path, size=256, interp='bilinear'):
    imgs = []

    for i, name in enumerate(image_names):

        im = misc.imread(path[i] + name)
        im = misc.imresize(im, (size, size), interp=interp)

        imgs.append(im)

    return np.stack(imgs)


def remove_background(image, background):
    imgs = []
    for x in image:
        diff = np.abs(background - x)
        diff  = ((diff > 50) * 255.).astype(np.int8)
        imgs.append(diff)
    return np.stack(imgs)
    


def seg_pre_process(image_names):

    x = pre_process(image_names, [ORIGIN_PATH]*image_names.shape[0])
    clean_img = misc.imread('data/seg_data/background/0000112.jpg')
    clean_img = misc.imresize(clean_img, (256,256))
    x = remove_background(x, clean_img)

    y = pre_process(image_names, [ANNO_PATH]*image_names.shape[0], size=64, interp='nearest')

    return x, y


if __name__ == '__main__':
    make_image_train_dataset()
    load_image_train_dataset()

    # make_detacion_train_dataset()
    # load_detacion_train_dataset()

    make_fcn_train_dataset()
    load_fcn_train_dataset()
