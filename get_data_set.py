import numpy as np

positive = np.loadtxt(open("data/positive_train_112.csv","rb"), delimiter=",", skiprows=0)
negative = np.loadtxt(open("data/negative_train_112.csv","rb"), delimiter=",", skiprows=0)

# Concatenate positive samples and negative samples
data = np.zeros((positive.shape[0]+negative.shape[0],positive.shape[1]))
data[:positive.shape[0],:] = positive
data[positive.shape[0]:,:] = negative

# Shuffle
np.random.shuffle(data)

np.savetxt("data/train_112.csv", data, delimiter=',')