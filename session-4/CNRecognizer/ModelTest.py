'''
Created on Jun. 2, 2019

@author: mohammedmostafa
'''
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.models import load_model
from Util.ModelTest import ModelTest

small_data_folder = "../data/"

#default sizes
Image_Width = 100
Image_Height = 100
Image_Depth = 3
targetSize = (Image_Width,Image_Height)

model = load_model("../model/CNNDetector_5_enr.h5")
model.summary()

#define the sub folders for both training and test
validate = os.path.join(small_data_folder,'test')

#now the easiest way to load data is to use the ImageDataGenerator
test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)

test_generator = test_data_generator.flow_from_directory(validate,
                                                         batch_size=1,
                                                         target_size=targetSize,
                                                         shuffle=False)

modelTest = ModelTest(generator=test_generator, model=model)
modelTest.test()