'''
Created on Dec. 4, 2018

@author: mohammedmostafa
'''
import numpy as np
from tensorflow.keras.applications import xception
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix,classification_report
import os

x_train = []
y_train = []
y_labels = {}

small_data_folder = "../data/"

#default sizes
Image_Width = 100
Image_Height = 100
Image_Depth = 3
targetSize = (Image_Width,Image_Height)
targetSize_withdepth = (Image_Width,Image_Height,Image_Depth)

model = load_model("../model/CNDetector_transfer.h5")
model.summary()

#define the sub folders for both training and test
validate = os.path.join(small_data_folder,'test')

#now the easiest way to load data is to use the ImageDataGenerator
test_data_generator = ImageDataGenerator(preprocessing_function=xception.preprocess_input)

test_generator = test_data_generator.flow_from_directory(validate,
                                                         batch_size=1,
                                                         target_size=targetSize,
                                                         shuffle=False)

    
x_train = np.array(x_train)
print(y_labels)
#normalize the image data same way  as the model
x_train = xception.preprocess_input(x_train)

# NOW WE LOAD THE PRE_TRAINED MODEL
FEATURE_EXTRACTOR = xception.Xception(weights='imagenet',include_top=False,input_shape=targetSize_withdepth)

#now we can extract the feature and save them for our images
features_x = FEATURE_EXTRACTOR.predict_generator(test_generator)

model = load_model("../model/CNDetector_transfer.h5")

predictions = model.predict(features_x)

row_index = predictions.argmax(axis=1)

filenames = test_generator.filenames
nb_samples = len(filenames)
y_true = test_generator.classes
target_names = test_generator.class_indices.keys()
print(target_names)
print(confusion_matrix(y_true, row_index))
print('Classification Report')
target_names = test_generator.class_indices.keys()
print(classification_report(test_generator.classes, row_index, target_names=target_names))






