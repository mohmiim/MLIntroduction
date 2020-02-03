'''
Created on Feb 2, 2020

@author: miim
'''


from dataPrep import loadData,sequenceData, plot_series
from tensorflow.keras.layers import LSTM,Conv1D,Dense,Lambda,Flatten
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber
from math import sqrt
from sklearn.metrics import mean_squared_error
from scipy.ndimage.interpolation import shift
import numpy as np

LSTM_NODES  =50

EPOCHS = 1000 
BATCH_SIZE = 200
DATA_FILE = "data/Sunspots.csv"

#time_train, train, time_valid, valid = loadData("data/Sunspots.csv",3000, 0, 2)
time_train, train, time_valid, valid = loadData(DATA_FILE,3000, 0, 2)


window_size = 60
shuffle_buffer_size = 1000

NN = False
LSTM_MODEL = True

print(time_train.shape)
print(train.shape)
print(time_valid.shape)
print(valid.shape)

X_train, y_train = sequenceData(train,window_size)
print(X_train.shape)
print(y_train.shape)


def createNNModel() :
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_dim=window_size))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam',metrics=["mae"])
    model.summary()
    return model

def fit(model, X_train, y_train, expand=False):
    if expand:
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    return model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE )

def create_CONV_LSTM_modle(node_count=64):
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(10, activation="relu"))
    model.add(Dense(1))
    model.add(Lambda(lambda x: x * 400))
    model.compile(loss=Huber(),optimizer=SGD(lr=1e-8, momentum=0.9),metrics=["mae"])
    return model

def create_LSTM_model( node_count = 50):   
    # define model
    model = Sequential()
    model.add(LSTM(node_count, activation='relu', input_shape=(window_size, 1),return_sequences=True))
    model.add(LSTM(node_count))
    model.add(Dense(node_count, activation='relu'))
    model.add(Dense(1))
    model.add(Lambda(lambda x: x * 400))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model

def shift_nested_array_value(source, replace_val= 0):
    index = 0 
    for val in source :
        if (index==len(source)-1):
            source[index] = [replace_val]
        else :
            source[index] = source[index+1]
        index = index+1
        
def validate(model,start,test, expand=False):
    forecast = list()
    current = start
    if expand :
        current = current.reshape((current.shape[0], current.shape[1], 1))
    for i in range(len(test)):
        y = model.predict(current)
        forecast.append(y[0][0])
        if expand:
            shift_nested_array_value(current[0], y[0][0])
        else :
            current[0] = shift(current[0], -1, cval=y)

    rmse = sqrt(mean_squared_error(test, forecast)) 
    print(' error = %.2f' % rmse)
    return rmse, forecast

if NN:
    model = createNNModel()
    fit(model,X_train, y_train )
elif LSTM_MODEL :
    model = create_CONV_LSTM_modle(50)
    fit(model,X_train, y_train, expand=True )
    
    
    

error, forecast = validate(model,X_train[-1:,:], valid, NN!=True)
plot_series(time_valid, valid, show = False)
plot_series(time_valid, forecast)




