# copyright 2018 quisutdeus7 all reserved
import numpy as np
import tensorflow as tf
import pandas as pd

# 격자수, 경계 박스수, class 종류(peptide, cylinder)
def get_pic_info(Sector=5, n_box=2, n_class=2):
    _sector=Sector
    _n_box=n_box
    _n_class=n_class
    return _sector,_n_box, _n_class

# 데이터 전처리에서 넘어온 5*5*12 데이터들을 
# def load_from_data():
'''ㅔ
def loss(data):
    S, B, C = get_pic_info()
    SS = S*S # number of grid cells

    # placeholder phase
    box_X = tf.placeholder(tf.float32, [None, 2])
    box_y = tf.placeholder(tf.float32, [None, 2])
'''

# kairos/detection_model.py 의 inner function로 넘겨줘야함.
def creat_batch():
    meta = pd.read_hdf("data.h5") # h5 파일 경로 edit

    # meta data shape : filename, width, height, object(name, xmin, ymin, xmax, ymax)
    # 1 line 1 object
    f_name = meta['filename'].tolist()  # filename
    w, h = meta['width'].tolist(), meta['height'].tolist() # box width, height
    obj = [meta[x].tolist() for x in ['name','xmin','ymin','xmax','ymax']] # object name, X/Y min, max values

    # regression 타겟을 계산
    cellx = 1. * w/S
    celly = 1. * h/S

    for i in range(len(obj)):
        centerx = .5*(obj[1][i] + obj[3][i])
        centery = .5*(obj[2][i] + obj[4][i])
        cx = centerx / cellx
        cy = centery / celly
        if cs >= S or cy >= S: # cs,cy가 S 범위 초과할 시
            return None, None
        
        

    # placeholder values
    probs = np.zeros([SS,C]) # class별 확률
    confs = np.zeros([SS,B]) # 물체가 있을 확률
    coord = np.zeros([SS,B,4]) # x,y,w,h
    proid = np.zeros([SS,C]) # 
    prear = np.zeros([SS,4]) # 



        
