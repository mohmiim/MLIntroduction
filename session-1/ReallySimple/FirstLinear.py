'''
Created on May 30, 2019

@author: mohammedmostafa
'''

import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt


model = Sequential();
model.add(Dense(units=1,input_dim=1))
model.compile(optimizer="sgd", loss = "mean_squared_error")

# simple function y = 2X - 3
xs = np.random.choice(np.arange(-3,3,.01),500)
ys = xs*2 -3 

x_test=np.linspace(-3,3,500)
y_test = x_test*2 -3 

model.summary()

model.fit(xs, ys,epochs=40)

print(model.predict([10.0]))

weights = model.layers[0].get_weights()[0]
bias = model.layers[0].get_weights()[1]
print("W1 = {0}".format(weights))
print("b = {0}".format(bias))

results = model.predict(x_test )

plt.plot(x_test,y_test,c='b')
plt.plot(x_test,results,c='r')
plt.show()

