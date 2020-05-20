
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


dataset_train = pd.read_csv("train.csv")
dataset_test = pd.read_csv("test.csv")

X_train = dataset_train.iloc[:, 1:].values
y_train = dataset_train.iloc[:, 0:1].values
X_test = dataset_test.iloc[:, 0:].values


'''
dataset = pd.read_csv("train.csv");

X = dataset.iloc[:, 1:].values
y = dataset.iloc[:, 0:1].values
'''
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_y = LabelEncoder()
y_train[:, 0] = labelencoder_y.fit_transform(y_train[:, 0])
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()

'''
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = 0.2)
'''


X_train = X_train/255
X_test = X_test/255

'''
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

img = X_train[1].reshape(28,28)
plt.imshow(img)
'''
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout

classifier = Sequential()

classifier.add(Dense(units = 512, kernel_initializer='uniform',  activation='relu', input_dim = 784))
classifier.add(Dropout(0.3))

classifier.add(Dense(units = 512, kernel_initializer='uniform',  activation='relu'))
classifier.add(Dropout(0.3))

classifier.add(Dense(units = 512, kernel_initializer='uniform',  activation='relu'))
classifier.add(Dropout(0.2))



classifier.add(Dense(units = 10, kernel_initializer='uniform',  activation='softmax'))

classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.models import load_model
fit = True
if fit == True:
    classifier.fit(X_train, y_train, batch_size = 16 , epochs = 50)
    classifier.save('dr_ann_model_2.h5')
else:
    classifier = load_model('dr_ann_model_2.h5')
    
y_pred = classifier.predict(X_test)
y_pred = onehotencoder.inverse_transform(y_pred)
#y_test = onehotencoder.inverse_transform(y_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_test, y_pred)

res = []
for i in range(0, y_pred.shape[0]):
    temp = []
    temp.append(i+1)
    temp.append(y_pred[i])
    res.append(temp)
    
res = np.array(res)

y_pred[:, 1:2] = y_pred[:,0:1]
np.savetxt("submission1.csv", res, fmt='%i', delimiter=",")

for i in range(0,5):
    img = X_test[i].reshape(28,28)
    plt.imshow(img)

d1 = pd.read_csv("submission.csv")
y_pred1 = d1.iloc[:, 1:2].values

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_pred, y_pred1)

from sklearn.metrics import accuracy_score
ac = accuracy_score(y_pred, y_pred1)
