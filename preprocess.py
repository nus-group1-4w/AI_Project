import cv2
import random
from settings import *

def data_preprocess(img):
    """
    Random crop

    """

    lamda = img.shape[0] / img.shape[1]
    if lamda <= threshold and lamda >= 1./threshold:
        return cv2.resize(img,(crop_size,crop_size))
    else:
        coor = random.randint(0, abs(img.shape[0]-img.shape[1]))
        if img.shape[0] > img.shape[1]:
            cropped_img = img[coor:coor+img.shape[1]+1][0:img.shape[1]+1]
        else:
            cropped_img = img[0:img.shape[0]+1][coor:coor+img.shape[0]+1]
        return cv2.resize(cropped_img,(crop_size,crop_size))