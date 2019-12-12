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

    

print(y_labels)


# NOW WE LOAD THE PRE_TRAINED MODEL
model0 = load_model("../model/model.h5")
features_x = model0.predict_generator(test_generator)
print(features_x.shape)

model1 = load_model("../model/model1.h5")
features_x1 = model1.predict_generator(test_generator)
print(features_x1.shape)

model2 = load_model("../model/model2.h5")
features_x2 = model2.predict_generator(test_generator)
print(features_x2.shape)

model3 = load_model("../model/model3.h5")
features_x3 = model3.predict_generator(test_generator)
print(features_x3.shape)

model4 = load_model("../model/model4.h5")
features_x4 = model4.predict_generator(test_generator)
print(features_x4.shape)

all_features = np.concatenate((features_x, features_x1,features_x2,features_x3,features_x4), axis=1)
model = load_model("../model/CNDetector_transfer.h5")

predictions = model.predict(all_features)

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






