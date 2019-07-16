'''
Created on Dec. 3, 2018

@author: mohammedmostafa
'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Flatten
from sklearn.externals import joblib
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from Util.StopCallBack import myCallBacks
from tensorflow.nn import relu

epochs = 200
#load the saved features
x_train = joblib.load("../data/x_train.dat")
y_train = joblib.load("../data/y_train.dat")
y_labels = joblib.load("../data/y_labels.dat")
y_train = to_categorical(y_train)
print(y_train)

model = Sequential()

#add our layers
model.add(Flatten(input_shape=x_train.shape[1:]))
model.add(Dense(128,activation=relu))
model.add(Dropout(0.2))
model.add(Dense(128,activation=relu))
model.add(Dropout(0.5))
model.add(Dense(len(y_labels),activation='softmax'))
history = model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])

model.summary()

callBack = myCallBacks()
model.fit(x_train,y_train,epochs=epochs,shuffle=True,verbose=2,callbacks=[callBack])


model.save("../model/CNDetector_transfer.h5")


