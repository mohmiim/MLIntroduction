'''
Created on May 30, 2019

@author: mohammedmostaqfa
'''
import numpy as np
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt

xs = np.random.choice(np.arange(-3,3,.01),500)
ys = xs**2

x_test=np.linspace(-3,3,1000)
y_test=x_test**2

model = Sequential();
model.add(Dense(units=1,input_dim=1))


model.compile(optimizer="sgd", loss = "mean_squared_error")
model.summary()
history = model.fit(xs, ys,epochs =400,  verbose=1) 

results = model.predict(x_test)

plt.plot(x_test,results,c='r')
plt.plot(x_test,y_test,c='b')
plt.show()
