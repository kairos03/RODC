# Copyright 2018 kairos03. All Right Reserved.

"""
Image data process (labeling)
"""
import os

from scipy import misc
import numpy as np
import h5py

DATA_PATH = 'data/'
PEPTIDE_PATH = DATA_PATH+'peptide/'
PARTICLE_PATH = DATA_PATH+'particle/'
IMAGE_TRAIN_DATASET_PATH = DATA_PATH + 'image_train.h5'
IMAGE_PATH = DATA_PATH + 'image.pkl'
LABLE_PATH = DATA_PATH + 'lable.pkl'

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

    # make lables
    zero = np.zeros((450, 1))
    one = np.ones((450, 1))
    peptide_lables = np.concatenate([one, zero], axis=1)
    particle_lables = np.concatenate([zero, one], axis=1)
    lables = np.concatenate([peptide_lables, particle_lables])
    assert lables.shape == (900, 2)
    assert (lables[449] == np.array([1, 0])).all() 
    assert (lables[450] == np.array([0, 1])).all() 

    # save
    with h5py.File(path, 'w') as hf:
        hf.create_dataset("images", data=imgs, compression="gzip", compression_opts=5)
        hf.create_dataset("lables", data=lables, compression="gzip", compression_opts=5)
    print("SAVED", path)


def load_image_train_dataset(path=IMAGE_TRAIN_DATASET_PATH):
    with h5py.File(path, 'r') as hf:
        images = hf.get("images")[:]
        lables = hf.get("lables")[:]
    print("LOADED", path, images.shape, lables.shape)

    return images, lables


if __name__ == '__main__':
    make_image_train_dataset()
    load_image_train_dataset()
