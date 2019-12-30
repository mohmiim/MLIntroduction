'''
Created on Dec 23, 2019

@author: mohammedmostafa
'''

import tensorflow as tf

modelPath = "../model/CNDetector_5.h5"
converter = tf.lite.TFLiteConverter.from_keras_model_file(modelPath)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
lite_model = converter.convert()
open("../model/CNDetector_Lite_5.tflite", "wb").write(lite_model)



