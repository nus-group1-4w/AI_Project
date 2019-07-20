import numpy as np
from sklearn import svm
from sklearn.externals import joblib

# Load dataset
train_set = np.loadtxt(open("data/train_112.csv","rb"), delimiter=",", skiprows=0)
test_set = np.loadtxt(open("data/test_112.csv","rb"), delimiter=",", skiprows=0)

# Split samples and labels
x_train = train_set[:,:train_set.shape[1]-1]
y_train = train_set[:,-1]
x_test = test_set[:,:test_set.shape[1]-1]
y_test = test_set[:,-1]

# Train the classifier
clf = svm.SVC(kernel='linear')
clf.fit(x_train,y_train)
score_train = clf.score(x_train,y_train)
score_test = clf.score(x_test,y_test)
print('linear:', score_train, score_test)

# Save the model
joblib.dump(clf, "model/train_model.m")