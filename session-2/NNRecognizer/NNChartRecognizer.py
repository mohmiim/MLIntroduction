'''
Created on Jun. 2, 2019

@author: mohammedmostafa
'''
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.nn import relu,softmax
from tensorflow.keras.optimizers import SGD
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
CLASSES_COUINT = 5;
epochs = 500

#define the sub folders for both training and test
training = os.path.join(small_data_folder,'train')

#now the easiest way to load data is to use the ImageDataGenerator
train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)


train_generator = train_data_generator.flow_from_directory(training,
                                                           batch_size=20,
                                                           target_size=targetSize,
                                                           #seed=12
                                                           )

model = Sequential()
model.add(Flatten(input_shape=targetSize_withdepth))
model.add(Dense(1024,activation=relu))
model.add(Dense(512,activation=relu))
model.add(Dense(CLASSES_COUINT,activation=softmax))

history = model.compile(optimizer=SGD(),loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

callBack = myCallBacks()

model.fit_generator(generator=train_generator,epochs=epochs,callbacks=[callBack])

model.save("../model/nnDetector_5c.h5")

