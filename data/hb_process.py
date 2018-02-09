import os
import cv2
import numpy as np

def get_red_yellow(labimg):
    lab_red = labimg[:, :, 1] # +: Red -: Green
    lab_yel = labimg[:, :, 2] # +: Yellow -: Blue
    
    lab_yel[lab_yel < 0] = 0 # Ignore blue
    
    # Green pixels in yellow channel are compensated by minus values in red channel
    ryimg = lab_red + lab_yel 
    ryimg[ryimg < 0] = 0
    
    # Truncating
    lab_red[lab_red < 0] = 0

    lab_onlyred = lab_red - lab_yel
    lab_onlyred[lab_onlyred < 0] = 0
    
    lab_onlyyel = ryimg - lab_onlyred
    lab_onlyyel[lab_onlyyel < 0] = 0
 
    lab_onlyyel /= np.amax(lab_onlyyel)
    lab_onlyred /= np.amax(lab_onlyred)
    ryimg /= np.amax(ryimg)

    return [lab_onlyred, lab_onlyyel, ryimg]


def get_clean_bw(grayimg, binarythreshold=2):
    kernel = np.ones((5, 5), np.uint8)

    opening = cv2.morphologyEx(grayimg, cv2.MORPH_OPEN, kernel)*5
    gradient = cv2.morphologyEx(opening, cv2.MORPH_GRADIENT, kernel)

    ret, thresh1 = cv2.threshold(gradient*255, binarythreshold, 255, cv2.THRESH_BINARY)
    closing = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)

    closing = closing * 255
    closing = closing.astype(np.uint8)

    im_floodfill = closing.copy()
     
    h, w = closing.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
     
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    im_out = closing | im_floodfill_inv

    mask = cv2.erode(im_out, kernel, iterations = 1)

    return cv2.medianBlur(mask, 5)


def get_roi(img, cleanimg):
    imgdiff1 = abs(cleanimg[:,:,0] - img[:,:,0])
    imgdiff2 = abs(cleanimg[:,:,1] - img[:,:,1])
    imgdiff3 = abs(cleanimg[:,:,2] - img[:,:,2])

    diff = imgdiff1 + imgdiff2 + imgdiff3
    diff = 2 * pow(diff, 2)
    
    diff[diff < 0.1] = 0
    diff[diff >= 0.1] = 1

    return get_clean_bw(diff)
    

def cvtColor(img, flag):
    if img.dtype == np.float and flag == cv2.COLOR_GRAY2BGR:
        return cv2.cvtColor((img*255).astype(np.uint8), flag)
    else:
        return cv2.cvtColor(img, flag)


def astype(img, type):
    if type == np.float32 and img.dtype == np.uint8:
        return img.astype(np.float32) / 255
    elif type == np.uint8 and img.dtype == np.float32:
        return (img * 255).astype(np.uint8)
    else:
        return img.astype(type)


def get_nofcomp(bwimg, sizethreshold):
    # num_labels, labels, stats, centroids. connectivity = 4
    
    stats = cv2.connectedComponentsWithStats(bwimg, 4, cv2.CV_8U)[2]

    count = [stats[i, cv2.CC_STAT_AREA] > sizethreshold for i in range(1, len(stats))]

    return sum(count)


def get_two_largest_comp(bwimg):
    stats = cv2.connectedComponentsWithStats(bwimg, 4, cv2.CV_8U)[2]
    areas = stats[1:, cv2.CC_STAT_AREA]
    areas.sort(reverse=True)

    return areas

def main():

    clean_front = astype(cv2.imread('cam1/clean.jpg'), np.float32)
    clean_right = astype(cv2.imread('cam2/clean.jpg'), np.float32)

    filenames = os.listdir('cam1')

    cv2.namedWindow('Result')

    filecount = 0

    for filename in filenames:
        if filename.find('.jpg') < 0:
            continue

        filecount += 1

        if filecount < 200:
            continue

        img_front = astype(cv2.imread('cam1/'+filename), np.float32)
        img_right = astype(cv2.imread('cam2/'+filename), np.float32)
        
        mask_front = get_roi(img_front, clean_front)
        mask_right = get_roi(img_right, clean_right)

        print(filename, end=' ')

        count_front = get_nofcomp(mask_front, 6000)
        count_right = get_nofcomp(mask_right, 6000)
        
        roi_front = cv2.bitwise_or(img_front, img_front, mask=mask_front)
        roi_right = cv2.bitwise_or(img_right, img_right, mask=mask_right)
        
        lab_front = cvtColor(roi_front, cv2.COLOR_BGR2LAB)
        lab_right = cvtColor(roi_right, cv2.COLOR_BGR2LAB)

        # float32 images
        [red_front, yel_front, ry_front] = get_red_yellow(lab_front)
        [red_right, yel_right, ry_right] = get_red_yellow(lab_right)
        
        red_front = get_clean_bw(red_front)
        yel_front = get_clean_bw(yel_front, 70)
        ry_front  = get_clean_bw( ry_front, 20)
        red_right = get_clean_bw(red_right)
        yel_right = get_clean_bw(yel_right, 70)
        ry_right  = get_clean_bw( ry_right, 20)

        yel_front = cv2.subtract(yel_front,red_front)
        yel_right = cv2.subtract(yel_right,red_right)

        sizethreshold = 1000

        count_rf  = get_nofcomp(red_front, sizethreshold)
        count_yf  = get_nofcomp(yel_front, sizethreshold)
        count_ryf = get_nofcomp(ry_front,  sizethreshold)
        count_rr  = get_nofcomp(red_right, sizethreshold)
        count_yr  = get_nofcomp(yel_right, sizethreshold)
        count_ryr = get_nofcomp(ry_right,  sizethreshold)

        # Only for visualization
        red_front = cvtColor(red_front, cv2.COLOR_GRAY2BGR)
        yel_front = cvtColor(yel_front, cv2.COLOR_GRAY2BGR)
        ry_front  = cvtColor( ry_front, cv2.COLOR_GRAY2BGR)
        red_right = cvtColor(red_right, cv2.COLOR_GRAY2BGR)
        yel_right = cvtColor(yel_right, cv2.COLOR_GRAY2BGR)
        ry_right  = cvtColor( ry_right, cv2.COLOR_GRAY2BGR)

        img1 = np.hstack((img_front, red_front, yel_front, ry_front))
        img2 = np.hstack((img_right, red_right, yel_right, ry_right))

        # img1 = np.hstack((red_front, yel_front, ry_front))
        # img2 = np.hstack((red_right, yel_right, ry_right))

        #ret, red_front = cv2.threshold(red_front*255, 2, 255, cv2.THRESH_BINARY)
        #ret, yel_front = cv2.threshold(yel_front*255, 2, 255, cv2.THRESH_BINARY)

        img = np.vstack((img1, img2))
        #img = np.vstack((mask_front, mask_right))
        #img = np.vstack((roi_front, roi_right))
        img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)

        print(str(count_front) + ' ' + str(count_right), end=' ')
        print(str(count_rf) + ' ' + str(count_rr) + ' ' + str(count_yf), end=' ')
        print(str(count_yr) + ' ' + str(count_ryf) + ' ' + str(count_ryr))

        # Possible contact => 1 1 1 1 1 1 1 1
        # Separate check => Any of numbers is 2
        # Single check => One or more 0, 1 otherwise

        cv2.imshow('Result', img)
        cv2.waitKey()

    cv2.destroyAllWindows()