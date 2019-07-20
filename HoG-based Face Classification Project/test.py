import glob
import cv2
from sklearn.externals import joblib
from settings import *
from hog import *
import preprocess

def evaluate(files):
    result = []
    clf = joblib.load('model/train_model.m')
    for path in files:
        img = cv2.imread(path,0)
        img = preprocess.data_preprocess(img)
        hog = HoG(img)
        feature_vector = hog.feature_extract()
        feature = np.array(feature_vector).flatten().reshape((1,feature_length))
        label = clf.predict(feature)
        result.append(int(label[0]))
    return result
