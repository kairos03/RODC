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
import cv2


# from data.hb_process import get_roi
# from data.hb_process import get_nofcomp
# from data.hb_process import cvtColor


DATA_PATH = 'data/seg_data/'
DATA_PATH2 = 'data/realimages/'

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

ORIGIN_PATH = DATA_PATH + 'images/'
MASK_PATH = DATA_PATH + 'labels/'
ANNO_PATH = DATA_PATH + 'annos/'
FCN_TRAIN_DATASET_PATH = DATA_PATH + 'fcn_train.h5'
CLASSFICATION_TRAIN_PATH = DATA_PATH2 + 'classification_train.h5'


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

def make_classification_dataset(path=CLASSFICATION_TRAIN_PATH):
    df = pd.DataFrame(columns=['filenames', 'is_contacted'])

    c_images, c_filenames = concatenated_image(DATA_PATH2 + 'contact/')
    s_images, s_filenames = concatenated_image(DATA_PATH2 + 'separate/')
    n_c_img = c_images.shape[0]//2
    n_s_img = s_images.shape[0]//2

    # contact + separate image
    # images = np.concatenate((c_images, s_images))
    filenames = np.concatenate((c_filenames, s_filenames))
    # contact + separate labels
    labels = np.concatenate((np.ones(n_c_img, ), np.zeros(n_s_img,)))

    print(filenames, filenames.shape, labels, labels.shape)
    df['filenames'] = filenames
    df['is_contacted'] = labels

    # save
    df.to_hdf(path, 'classification')
    print("SAVED", path, df.shape)

def concatenated_image(path):
    """contact front and right data""" 
    # read images
    f_fnames, f_images = read_image(path+'front/')
    r_fnames, r_images = read_image(path+'right/')

    #for r in range(r_fnames.shape[0]):
    #    for f in range(f_fnames.shape[0]):
    #        if r_fnames[r] == f_fnames[f]:
    #            break
    #        elif f == (f_fnames.shape[0]-1):
    #            print(r_fnames[r])
    #            print(r_fnames.shape)
                
    # concat two images
    images = np.concatenate((f_images, r_images), 0) # f,f,f,...,r,r,,,,

    return images, f_fnames

def read_image(path):
    """ Read images from path and return fnames, images"""
    print(os.listdir('.'))
    print(path)
    fnames = np.array(os.listdir(path))
    fnames.sort()
    # resize_path = path.replace(DATA_PATH2, DATA_PATH2 + 'resize/')

    imgs = []
    for name in fnames:
        img = misc.imread(path+name)
        # print(img.shape, path+name)
        img = misc.imresize(img, (256, 256))
        imgs.append(img)
    #    misc.imsave(resize_path+name, img)
    print(img, img.shape)
    imgs = np.stack(imgs)
    print(img, img.shape)
    return fnames, imgs

def load_classification_dataset(path=CLASSFICATION_TRAIN_PATH):
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
        mask = np.abs(background - x)
        mask  = (mask < 25)
        np.place(x, mask, 255)
        imgs.append(x)
    return np.stack(imgs)
    

def seg_pre_process(image_names):

    images = pre_process(image_names, [ORIGIN_PATH]*image_names.shape[0])
    clean_img = misc.imread('data/seg_data/clean.jpg')
    clean_img = misc.imresize(clean_img, (256,256))

    x = []
    for img in images:
        mask = get_roi(img, clean_img)
        roi = cv2.bitwise_or(img, img, mask=mask)
        # lab = cvtColor(mask, cv2.COLOR_GRAY2RGB)
        x.append(roi)
    
    x = np.stack(x)
    y = pre_process(image_names, [ANNO_PATH]*image_names.shape[0], size=128, interp='nearest')

    return x, y
    

if __name__ == '__main__':
    # make_image_train_dataset()
    # load_image_train_dataset()

    # make_detacion_train_dataset()
    # load_detacion_train_dataset()

    # before deconvenet_train
    # make_fcn_train_dataset()
    # load_fcn_train_dataset()

    make_classification_dataset()
    load_classification_dataset()
