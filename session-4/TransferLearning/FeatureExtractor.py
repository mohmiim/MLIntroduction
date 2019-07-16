'''
Created on Jun. 5, 2019

@author: mohammedmostafa
'''

from sklearn.externals import joblib
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import xception


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
FEATURE_EXTRACTOR = xception.Xception(weights='imagenet',include_top=False,input_shape=targetSize_withdepth)

#now we can extract the feature and save them for our images
features_x = FEATURE_EXTRACTOR.predict_generator(train_generator)

#we can save it now 
joblib.dump(features_x,"../data/x_train.dat")
joblib.dump(y_train,"../data/y_train.dat")
joblib.dump(y_labels,"../data/y_labels.dat")






   



    