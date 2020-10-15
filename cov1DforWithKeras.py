# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 21:16:48 2020

@author: shuntarou
"""

from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import keras
from keras import layers
from keras import models
from keras.layers import Dense
from keras import optimizers
from keras.models import load_model	
from keras.layers import Conv2D, Conv1D
from keras.layers import MaxPooling2D, MaxPooling1D, Flatten, GlobalAveragePooling1D, Dropout
from keras.utils import to_categorical
import matplotlib.pyplot as plt

dataName = 'survival-data0428.csv' #a data name for analysis
df = pd.read_csv(dataName, encoding="CP932")

df = df.dropna(how='all')

inputNames = [] #column name of inputs

y_output = ""

df_train = df[:(round(0.8*len(df)))]
df_test = df[(round(0.8*len(df))):]


nCol =  len(pd.get_dummies(df[inputNames]).columns)
dummy_data = pd.get_dummies(df[inputNames]).values

x_train = dummy_data[:(round(0.8*len(df)))]
y_train = df_train[y_output].values

x_test = dummy_data[(round(0.8*len(df))):]
y_test = df_test[y_output].values

x_train = x_train.reshape(x_train.shape[0], 1, nCol)
y_train = y_train.reshape(y_train.shape[0], 1)

x_test = x_test.reshape(x_test.shape[0], 1, nCol)
y_test = y_test.reshape(y_test.shape[0], 1)

y_train_mean = y_train.mean()
y_train_std = y_train.std()
y_train -= y_train_mean
y_train /= y_train_std

y_test -= y_train_mean
y_test /= y_train_std

model = models.Sequential() 
model.add(Conv1D(64, 8, padding = 'same', activation='relu', input_shape=(1, nCol)))
model.add(Conv1D(64, 8, padding = 'same', activation='relu'))
model.add(MaxPooling1D(3, padding='same'))
model.add(Conv1D(128, 8, padding = 'same', activation='relu'))
model.add(Conv1D(128, 8, padding = 'same', activation='relu'))
model.add(GlobalAveragePooling1D())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae', 'mse'])
history = model.fit(x_train, y_train, batch_size = 20, epochs = 100,
                    validation_data = (x_test, y_test))

score = model.evaluate(x_train, y_train, verbose = 1)		
print('mae=', score[0], 'mse=', score[1])

model.summary()

#モデルの保存
model.save('model.h5')		
model = load_model('model.h5')		

#パラメータの保存
model.save_weights('param.hdf5')
model.load_weights('param.hdf5')

plt.plot(range(100), history.history['loss'], label = 'loss')
plt.plot(range(100), history.history['val_loss'], label = 'val_loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()
plt.show()