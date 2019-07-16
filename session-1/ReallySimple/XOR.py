'''
Created on May 30, 2019

@author: mohammedmostaqfa
'''

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential


xs = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=float)
ys = np.array([0,1,1,0],dtype=float)
xv = xs
yv = ys

model = Sequential();
model.add(Dense(units=32,input_dim=2, activation='relu'))
model.add(Dense(units=16,input_dim=2, activation='relu'))
model.add(Dense(units=1, activation='sigmoid'))


model.compile(optimizer="sgd", loss = "mean_squared_error",metrics=['acc'])

history = model.fit(xs, ys, epochs=1000,verbose=2) 

print(model.predict(xs).round())






