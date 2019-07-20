# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 08:50:54 2019

@author: 23899
"""
def rotate(image, angle=90, center=None, scale=1.0):
    (h, w) = image.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))
    return rotated


import os
import cv2
import os.path
imagelist = os.listdir('./train_data/train/p/')
for each in imagelist:
    filetype='.jpg'
    filename=each.replace(filetype,'')
    path=r"G:\HUST\MSE\Study Abroad\2019NUS_summer\Project\train_data\train\p"+'\\'+each
    img1 = cv2.imread(path)
    res1= rotate(img1)
    newname1=r"G:\HUST\MSE\Study Abroad\img"+'\\'+filename+"(1).jpg"
    cv2.imwrite(newname1, res1)
    res2=cv2.flip(img1, 1)
    newname2=r"G:\HUST\MSE\Study Abroad\img"+'\\'+filename+"(2).jpg"
    cv2.imwrite(newname2, res2)
    