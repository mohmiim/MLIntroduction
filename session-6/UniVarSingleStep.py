'''
Created on Feb 2, 2020

@author: miim
'''

from dataPrep import loadData,sequenceData, plot_series
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM,Conv1D,Dense,MaxPooling1D,Flatten,Dropout
from tensorflow.keras.models import Sequential
from math import sqrt
from sklearn.metrics import mean_squared_error
import numpy as np

time_train, train, time_valid, valid = loadData('data/Sunspots.csv',3000, 0, 2)
train_mean = train.mean()
train_std = train.std()

# constants
EPOCHS = 400
BATCH_SIZE = 100
window_size = 60
FILTERS_COUNT = 256
KERNER_SIZE = 5
NODES = 400

print(time_train.shape)
print(train.shape)
print(time_valid.shape)
print(valid.shape)

X_train, y_train = sequenceData(train,window_size)

X_train = (X_train-train_mean)/train_std
y_train = (y_train-train_mean)/train_std

print(X_train.shape)
print(y_train.shape)
print(train_mean)
print(train_std)

def createConvLstmModel() :
  model = Sequential()
  model.add(Conv1D(FILTERS_COUNT, KERNER_SIZE, activation='relu', input_shape=(window_size,1)))
  model.add(MaxPooling1D())
  model.add(Conv1D(FILTERS_COUNT, KERNER_SIZE, activation='relu', input_shape=(window_size,1)))
  model.add(Dropout(0.5))
  model.add(MaxPooling1D())
  model.add(LSTM(NODES,activation='relu'))
  model.add(Dense(1))
  model.compile(loss='mse', optimizer='adam')
  model.summary()
  return model

def fitConvLstm(X_train, y_train) :
  X_train = X_train.reshape((X_train.shape[0],X_train.shape[1], 1))
  print(X_train.shape)
  return model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)


def lstmValidation(model, train, test):
  predictions = list()
  history = [x for x in train]
  history = np.asarray(history)
  for i in range(len(test)):
    x_input = np.array(history[-window_size:]).reshape((1, window_size, 1))
    x_input = (x_input-train_mean)/train_std
    yhat = model.predict(x_input)[0]
    yhat = (yhat*train_std) + train_mean
    predictions.append(yhat)
    history = np.append(history, test[i])
  error  = sqrt(mean_squared_error(test, predictions)) 
  plot_series(range(0,len(test)), test, False)
  plot_series(range(0,len(test)), predictions)
  print(' > %.3f' % error)
  return error
  
model = createConvLstmModel()
fitConvLstm(X_train,y_train)

lstmValidation(model,train, valid)
  
  
  
  




