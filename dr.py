
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

'''
dataset_train = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")

X_train = dataset_train.iloc[:, 1:].values
y_train = dataset_train.iloc[:, 0:1].values

X_test = dataset_test.iloc[:, 0:].values

'''

dataset = pd.read_csv("train.csv");

X = dataset.iloc[0:2000, 1:].values
y = dataset.iloc[0:2000, 0:1].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2)

X_train = X_train/255
X_test = X_test/255

X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

img = X_train[1].reshape(28,28)
plt.imshow(img)

import keras
from keras.models import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D
from keras.layers import Dropout,Flatten

