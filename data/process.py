# Copyright 2018 kairos03. All Right Reserved.

"""
Image data process (labeling)
"""
import os
import pickle

from scipy import misc
import numpy as np

DATA_PATH = 'data/'
PEPTIDE_PATH = DATA_PATH+'peptide/'
PARTICLE_PATH = DATA_PATH+'particle/'
IMAGE_PATH = DATA_PATH + 'image.pkl'
LABLE_PATH = DATA_PATH + 'lable.pkl'

def image_to_pkl():
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
    assert len(imgs) == 900

    # save to pickle
    imgs = np.asarray(imgs)
    pickle.dump(imgs, open(IMAGE_PATH, 'wb'), protocol=4)
    print("SAVED", IMAGE_PATH, imgs.shape)

    # make lables
    zero = np.zeros((450, 1))
    one = np.ones((450, 1))
    peptide_lables = np.concatenate([one, zero], axis=1)
    particle_lables = np.concatenate([zero, one], axis=1)
    labels = np.concatenate([peptide_lables, particle_lables])
    assert labels.shape == (900, 2)
    assert (labels[449] == np.array([1, 0])).all() 
    assert (labels[450] == np.array([0, 1])).all() 

    pickle.dump(labels, open(LABLE_PATH, 'wb'), protocol=4)
    print("SAVED", LABLE_PATH, labels.shape)


def load_data(image_path=IMAGE_PATH, lable_path=LABLE_PATH):

    images = pickle.load(open(image_path, 'rb'))
    print("LOADED", image_path, images.shape)

    lables = pickle.load(open(lable_path, 'rb'))
    print("LOADED", lable_path, lables.shape)

    return images, lables


if __name__ == '__main__':
    image_to_pkl()
    load_data()
