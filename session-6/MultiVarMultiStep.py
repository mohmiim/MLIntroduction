'''
Created on Feb 2, 2020

@author: miim
'''


from dataPrep import loadData,sequenceData, plot_series
import pandas as pd
import matplotlib.pyplot as plt
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Conv1D,Dense,Lambda,Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import Huber
from math import sqrt

CSV_FILE = "data/combined_csv.csv"

df = pd.read_csv(CSV_FILE)


columnsToConsider = ['p (mbar)', 'T (degC)', 'rho (g/m**3)']
features = df[columnsToConsider]
features.index = df['Date Time']


features.plot(subplots=True)
plt.show()

TRAIN_DATA_SIZE = 256

dataset = features.values

data_mean = dataset[:TRAIN_DATA_SIZE].mean(axis=0)
data_std = dataset[:TRAIN_DATA_SIZE].std(axis=0)

dataset = (dataset-data_mean)/data_std

print(len(dataset))

training_data = dataset[:TRAIN_DATA_SIZE]
validation_data = dataset[TRAIN_DATA_SIZE:]

print(len(training_data))
print(len(validation_data))

print(data_mean)
print(data_std)

def sequenceData(dataset, target, start_index, end_index, history_size,target_size, step, single_step=False):
    X = []
    y = []
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        X.append(dataset[indices])
        if single_step:
            y.append(target[i+target_size])
        else:
            y.append(target[i:i+target_size])
    return np.array(X), np.array(y)


LOOK_AHEAD = 72
STEP = 6
WINDOW_SIZE = 720
BATCH_SIZE = 20000
BUFFER_SIZE = 1000




X_train, y_train = sequenceData(dataset, dataset[:, 1], 0,
                                 TRAIN_DATA_SIZE, WINDOW_SIZE,
                                 LOOK_AHEAD, STEP)

X_val, y_val = sequenceData(dataset, dataset[:, 1],
                            TRAIN_DATA_SIZE, None, WINDOW_SIZE,
                            LOOK_AHEAD, STEP)

print(X_train.shape)
print(y_train.shape)
print(X_val.shape)
print(y_val.shape)



train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_data = train_data.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_data = val_data.batch(BATCH_SIZE)

def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot(history, true_future, prediction):
    plt.figure(figsize=(12, 6))
    num_in = create_time_steps(len(history))
    num_out = len(true_future)
    plt.plot(num_in, np.array(history[:, 1]), label='History')
    plt.plot(np.arange(num_out)/STEP, np.array(true_future), 'bo',
             label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out)/STEP, np.array(prediction), 'ro',
                 label='Predicted Future')
        plt.legend(loc='upper left')
    plt.show()

for x, y in train_data.take(1):
    multi_step_plot(x[0], y[0], np.array([0]))

def createModel() :
    model = Sequential()
    model.add(LSTM(32,return_sequences=True,input_shape=X_train.shape[-2:]))
    model.add(LSTM(16, activation='relu'))
    model.add(Dense(72))
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='mae')
    return model

model  = createModel()
model.summary()

EVALUATION_INTERVAL = 200
EPOCHS = 10

history = model.fit(train_data, epochs=EPOCHS)

model.save("model/model.h5")

for x, y in val_data.take(3):
  multi_step_plot(x[0], y[0], multi_step_model.predict(x)[0])