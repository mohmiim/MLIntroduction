'''
Created on Jun. 5, 2019

@author: mohammedmostafa
'''

from sklearn.externals import joblib
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import xception, inception_v3, resnet50, vgg19,densenet
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential

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
x_train = []
y_train = []
y_labels = []

#define the sub folders for both training and test
training = os.path.join(small_data_folder,'train')


#now the easiest way to load data is to use the ImageDataGenerator
train_data_generator = ImageDataGenerator(preprocessing_function=xception.preprocess_input,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          zoom_range=0.2,
                                          fill_mode='nearest')


train_generator = train_data_generator.flow_from_directory(training,
                                                           batch_size=20,
                                                           target_size=targetSize,
                                                           #seed=12
                                                           shuffle=False
                                                           )




y_train =  train_generator.classes
for k in train_generator.class_indices.keys():
    y_labels.append(k)


print(len(y_train))

# NOW WE LOAD THE PRE_TRAINED MODEL
FEATURE_EXTRACTOR = vgg19.VGG19(weights='imagenet',include_top=False,input_shape=targetSize_withdepth)
model = Sequential()
model.add(FEATURE_EXTRACTOR)
model.add(Flatten())
features_x = model.predict_generator(train_generator)
print(type(features_x).__name__)
print(features_x.shape)
model.save("../model/model.h5",include_optimizer=False)

FEATURE_EXTRACTOR1 = xception.Xception(weights='imagenet',include_top=False,input_shape=targetSize_withdepth)
model1 = Sequential()
model1.add(FEATURE_EXTRACTOR1)
model1.add(Flatten())
features_x1 = model1.predict_generator(train_generator)
print(type(features_x1).__name__)
print(features_x1.shape)
model1.save("../model/model1.h5",include_optimizer=False)

FEATURE_EXTRACTOR2 = resnet50.ResNet50(weights='imagenet',include_top=False,input_shape=targetSize_withdepth)
model2 = Sequential()
model2.add(FEATURE_EXTRACTOR2)
model2.add(Flatten())
features_x2 = model2.predict_generator(train_generator)
print(type(features_x2).__name__)
print(features_x2.shape)
model2.save("../model/model2.h5",include_optimizer=False)

FEATURE_EXTRACTOR3 = inception_v3.InceptionV3(weights='imagenet',include_top=False,input_shape=targetSize_withdepth)
model3 = Sequential()
model3.add(FEATURE_EXTRACTOR3)
model3.add(Flatten())
features_x3 = model3.predict_generator(train_generator)
print(type(features_x3).__name__)
print(features_x3.shape)
model3.save("../model/model3.h5",include_optimizer=False)

FEATURE_EXTRACTOR4 = densenet.DenseNet201(weights='imagenet',include_top=False,input_shape=targetSize_withdepth)
model4 = Sequential()
model4.add(FEATURE_EXTRACTOR4)
model4.add(Flatten())
features_x4 = model4.predict_generator(train_generator)
print(type(features_x4).__name__)
print(features_x4.shape)
model4.save("../model/model4.h5",include_optimizer=False)



all_features = np.concatenate((features_x, features_x1,features_x2,features_x3,features_x4), axis=1)


print(type(all_features).__name__)
print(all_features.shape)


#we can save it now 
joblib.dump(all_features,"../data/x_train.dat")
joblib.dump(y_train,"../data/y_train.dat")
joblib.dump(y_labels,"../data/y_labels.dat")






   



    