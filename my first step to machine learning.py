from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import keras
from keras import layers
from keras import models
from keras.layers import Dense
from keras import optimizers
from keras.models import load_model	
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, Flatten
from keras.utils import to_categorical


dataName = '.csv' #a data name for analysis
df = pd.read_csv(dataName, encoding="CP932")

df[''] = df[''].astype(np.float64) #output
    
print(df)
colNames = [''] #column name of inputs

df_train = df[:(round(0.8*len(df)))]
df_test = df[(round(0.8*len(df))):]

dummy_data = pd.get_dummies(df[colNames]).values

x_train = dummy_data[:(round(0.8*len(df)))]
y_train = df_train[''].values

x_test = dummy_data[(round(0.8*len(df))):]
y_test = df_test[''].values

#x_train = x_train.reshape(x_train.shape[0], 1, 51, 1)
x_train = x_train.reshape(x_train.shape[0], 51)
#x_train = to_categorical(x_train)
y_train = y_train.reshape(y_train.shape[0], 1)

x_test = x_test.reshape(x_test.shape[0], 51)
y_test = y_test.reshape(y_test.shape[0], 1)


y_train_mean = y_train.mean()
y_train_std = y_train.std()
y_train -= y_train_mean
y_train /= y_train_std

y_test -= y_train_mean
y_test /= y_train_std

model = models.Sequential() 
#model.add(Conv2D(16, (1, 3), input_shape = (1, 51, 1), activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2, 2), padding = 'same',))
#model.add(Flatten())
#model.add(Dense(1, activation = 'softmax'))

model.add(Dense(64, activation ='relu', input_shape=(51, )))
model.add(Dense(64, activation ='relu'))
model.add(Dense(1))
model.compile(loss = 'mse', optimizer = 'adam', metrics = ['mae'])

model.fit(x_train, y_train, batch_size = 20, epochs = 1000, validation_data = (x_test, y_test))

score = model.evaluate(x_train, y_train, verbose = 1)		
print('mae=', score[0], 'mse=', score[1])

model.summary()

#モデルの保存
model.save('model.h5')		
model = load_model('model.h5')		

#パラメータの保存
model.save_weights('param.hdf5')
model.load_weights('param.hdf5')

to.csv()