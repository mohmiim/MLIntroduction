'''
Created on Jun. 2, 2019

@author: mohammedmostafa
'''

import os
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.nn import relu
from tensorflow.keras.optimizers import Adam
from Util.StopCallBack import myCallBacks

#first step is to load our data 

#define the root folder for the data 
small_data_folder = "../data/"

#default sizes
Image_Width = 100
Image_Height = 100
Image_Depth = 3
targetSize = (Image_Width,Image_Height)
targetSize_withdepth = (Image_Width,Image_Height,Image_Depth)

epochs = 500

#define the sub folders for both training and test
training = os.path.join(small_data_folder,'train')


#now the easiest way to load data is to use the ImageDataGenerator
train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          zoom_range=0.2,
                                          fill_mode='nearest')


train_generator = train_data_generator.flow_from_directory(training,
                                                           batch_size=20,
                                                           target_size=targetSize,
                                                           #seed=12
                                                           )



callBack = myCallBacks()

model = Sequential()
model.add(Conv2D(128,(3,3),input_shape=targetSize_withdepth,activation=relu))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(64,(3,3),activation=relu))
model.add(MaxPool2D((2,2)))
model.add(Conv2D(32,(3,3),activation=relu))
model.add(MaxPool2D((2,2)))
model.add(Flatten())
model.add(Dense(512,activation=relu))
model.add(Dense(2,activation='softmax'))

history = model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])


model.summary()


model.fit_generator(generator=train_generator,epochs=epochs,callbacks=[callBack])

model.save("../model/CNDetector.h5")

