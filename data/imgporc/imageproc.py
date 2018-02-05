#!/usr/bin/python

import cv2
import numpy as np

cleanimg = cv2.imread('clean.jpg', 0)
synimg   = cv2.imread('000007.jpg')
graysyn  = cv2.imread('000007.jpg', 0)

maskimg = abs(cleanimg - graysyn)
maskimg = cv2.medianBlur(maskimg, 25)
maskimg = maskimg > 10

synimg[:,:,0] = np.multiply(synimg[:,:,0], maskimg)
synimg[:,:,1] = np.multiply(synimg[:,:,1], maskimg)
synimg[:,:,2] = np.multiply(synimg[:,:,2], maskimg)

labsyn = cv2.cvtColor(synimg, cv2.COLOR_BGR2LAB)

def get_red_yellow(labimg):
    lab_red = labimg[:, :, 1]
    lab_yel = labimg[:, :, 2]
	
    lab_red = lab_red.astype(np.float)
    lab_yel = lab_yel.astype(np.float)
		
    lab_red -= 128
    lab_yel -= 128
    lab_yel[lab_yel < 0] = 0
	
    ryimg = lab_red + lab_yel
    ryimg /= np.amax(ryimg)

    ryimg[ryimg < 0] = 0
    ryimg[ryimg > 1] = 1
	
    return ryimg
	
ryimg = get_red_yellow(labsyn)

cv2.imwrite('ROI.jpg', synimg)
cv2.imwrite('result.jpg', ryimg*255)

