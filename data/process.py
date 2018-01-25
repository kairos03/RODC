# Copyright 2018 kairos03. All Right Reserved.

"""
Image data process (labeling)
"""
import os

import xml.etree.ElementTree as ET

from scipy import misc
import numpy as np
import pandas as pd
import h5py

DATA_PATH = 'data/'

PEPTIDE_PATH = DATA_PATH+'peptide/'
PARTICLE_PATH = DATA_PATH+'particle/'
IMAGE_TRAIN_DATASET_PATH = DATA_PATH + 'image_train.h5'

DETACTION_OBJECTS_PATH = DATA_PATH + 'objects/'
DETACTION_TRAIN_DATASET_PATH = DATA_PATH + 'detation_train.h5'

def make_image_train_dataset(path=IMAGE_TRAIN_DATASET_PATH):
    imgs = []
    
    # load peptide images
    for image_name in os.listdir(PEPTIDE_PATH):
        image = misc.imread(PEPTIDE_PATH + image_name)
        imgs.append(image)
    assert len(imgs) == 450
    
    # load particle images
    for image_name in os.listdir(PARTICLE_PATH):
        image = misc.imread(PARTICLE_PATH + image_name)
        imgs.append(image)
    imgs = np.asarray(imgs)
    assert imgs.shape[0] == 900

    # TODO Move to model
    # image crop 
    imgs = imgs[:, 2:, 100:610, :] # 510, 510
    print(imgs.shape)

    # image resize
    resize = []
    for i in range(imgs.shape[0]):
        img = misc.imresize(imgs[i,:,:,:], (320, 320))
        resize.append(img)
    resize = np.stack(resize)
    print(resize.shape)
    assert resize.shape == (900, 320, 320, 3)

    # make labels
    zero = np.zeros((450, 1))
    one = np.ones((450, 1))
    peptide_labels = np.concatenate([one, zero], axis=1)
    particle_labels = np.concatenate([zero, one], axis=1)
    labels = np.concatenate([peptide_labels, particle_labels])
    assert labels.shape == (900, 2)
    assert (labels[449] == np.array([1, 0])).all() 
    assert (labels[450] == np.array([0, 1])).all() 

    # save
    with h5py.File(path, 'w') as hf:
        hf.create_dataset("images", data=resize, compression="gzip", compression_opts=5)
        hf.create_dataset("labels", data=labels, compression="gzip", compression_opts=5)
    print("SAVED", path)


def load_image_train_dataset(path=IMAGE_TRAIN_DATASET_PATH):
    with h5py.File(path, 'r') as hf:
        images = hf.get("images")[:]
        labels = hf.get("labels")[:]
    print("LOADED", path, images.shape, labels.shape)

    return images, labels


def make_detacion_train_dataset(path=DETACTION_OBJECTS_PATH):
    annotaion_path = 'Annotation/'
    image_path = 'Image/'

    dataset = pd.DataFrame(columns=['x', 'y', 'w', 'h', 'image', 'filename'])

    for xml_path in os.listdir(DETACTION_OBJECTS_PATH+annotaion_path):
        
        # Annotaion XML parsing
        tree = ET.parse(DETACTION_OBJECTS_PATH+annotaion_path+xml_path)
        root = tree.getroot()

        # extract img_filename and coord 
        img_filename = root.find('filename').text
        bndbox = root.find('object/bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        size = root.find('size')
        width = int(size.find('width').text)
        height = int(size.find('height').text)

        # calculate center x, center y, w, h
        w = (xmax - xmin) 
        h = (ymax - ymin)
        x = (w/2) + xmin
        y = (h/2) + ymin 

        # range in [0, 1]
        w = w/width
        h = h/height
        x = x/width
        y = y/height

        # read image
        image = misc.imread(DETACTION_OBJECTS_PATH+image_path+img_filename)
        
        # append data
        dataset = dataset.append({'x': x, 'y': y, 'w': w, 'h': h, 'image': image, 'filename': img_filename}, ignore_index=True)
        
    # save 
    dataset.to_hdf(DETACTION_TRAIN_DATASET_PATH, 'detacion')
    # print(dataset.head)
    print("SAVED", DETACTION_TRAIN_DATASET_PATH)


def load_detacion_train_dataset(path=DETACTION_TRAIN_DATASET_PATH):
    df = pd.read_hdf(path)
    print("LOADED", path, df.shape)
    return df

if __name__ == '__main__':
    make_image_train_dataset()
    load_image_train_dataset()

    make_detacion_train_dataset()
    load_detacion_train_dataset()
