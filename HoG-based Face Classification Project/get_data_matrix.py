import os
import cv2
import numpy as np
import preprocess
from settings import *
from hog import *

"""
HoG para: cell_size = 8, bin_size = 8, block_size = 16, img_size = (112,112)
          ---> hog_vector_size: (img_width/cell_size-1)*(img_height/cell_size-1)*((block_size/cell_size)**2)*bin_size
                               = (112/8-1)*(112/8-1)*((16/8)**2)*8 = 5408

"""

img_list = os.listdir('orig/p_train/')
#img_list = os.listdir('orig/p_test/')
#img_list = os.listdir('orig/n_train/')
#img_list = os.listdir('orig/n_test/')

data = np.ones((len(img_list),feature_length+1))
row_ind = 0
for image in img_list:

    path = 'orig/p_train/' + image
    #path = 'orig/p_test/' + image
    #path = 'orig/n_train/' + image
    #path = 'orig/n_test/' + image

    img = cv2.imread(path,cv2.IMREAD_GRAYSCALE)
    img = preprocess.data_preprocess(img)
    hog = HoG(img)
    feature_vector = hog.feature_extract()
    feature = np.array(feature_vector).flatten()
    data[row_ind,:data.shape[1]-1] = feature
    row_ind += 1

np.savetxt("data/positive_train_112.csv", data, delimiter=',')
#np.savetxt("data/positive_test_112.csv", data, delimiter=',')
#np.savetxt("data/negative_train_112.csv", data, delimiter=',')
#np.savetxt("data/negative_test_112.csv", data, delimiter=',')