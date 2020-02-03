'''
Created on Feb 2, 2020

@author: miim
'''
import csv
import matplotlib.pyplot as plt
import numpy as np
from _ast import IsNot
from pandas import DataFrame,concat
import tensorflow as tf


def plot_series(time, series, show=True, format='-'):
    plt.plot(time, series)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    if show:
        plt.show()

def loadData(pathToFile,sampleCount=None, timeIndex = 0 , dataIndex=1):
    time = []
    data = []
    plt.figure(figsize=(10, 6))
    with open(pathToFile) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            data.append(float(row[dataIndex]))
            time.append(int(row[timeIndex]))
    
    series = np.array(data)
    time = np.array(time)
    if sampleCount != None :
        train, valid = split(sampleCount, series)
        ttrain, tvalid = split(sampleCount, time)
        plot_series(ttrain, train, show=False)
        plot_series(tvalid, valid)
        return ttrain, train , tvalid, valid
    else :
        plot_series(time, series)
        return time,series
    
def split(sampleCount, series):
    X_train = series[:sampleCount]
    X_valid = series[sampleCount:]
    return X_train, X_valid

def sequenceData(train,window_size, output=True, shuffle= True) :
    if output : window_size = window_size + 1
    dataset = tf.data.Dataset.from_tensor_slices(train)
    dataset = dataset.window(window_size, shift=1, drop_remainder=True)
    dataset = dataset.flat_map(lambda w: w.batch(window_size + 1))
    if shuffle :
        dataset = dataset.shuffle(1000)
    
    X = []
    y = []
    if output :
        dataset = dataset.map(lambda w: (w[:-1], w[-1:]))
        for a,b in dataset:
            X.append(a.numpy())
            y.append(b.numpy())
    else :
        for f in dataset :
            X .append(f.numpy())
    return np.asarray(X), np.asarray(y)
    
    