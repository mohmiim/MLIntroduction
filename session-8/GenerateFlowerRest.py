'''
Created on Dec. 4, 2018

@author: mohammedmostafa
'''
import numpy as np
from numpy import random, asarray
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import gridspec
from PIL import Image
from flask import Flask, request
from flask import render_template, send_file
import datetime
import io
import os

# creates a Flask application, named app
app = Flask(__name__)

FLOWER_MODEL_FILE = "model/flowerGAN.h5"
DIGITS_MODEL_FILE = "model/model_11.h5"

#loading our pre-trained models
flowerModel = load_model(FLOWER_MODEL_FILE)
digitsModel = load_model(DIGITS_MODEL_FILE)

flowerModel.summary()
digitsModel.summary()

LATENT_DIM = 100


def generate_latent_input(latentDim):
    X = random.randn(latentDim)
    X = X.reshape((1,latentDim))
    X = X.astype('float32')
    return X


def generateSampleOutput(generator,X):
    y = generator.predict([X])
    imageData = y[0]
    imageData = (imageData * 127.5 ) + 127.5
    return imageData



@app.route('/generateFlower', methods=['GET'])
def generateFlower():
    X = generate_latent_input(LATENT_DIM)
    # generate images
    imageData =  generateSampleOutput(flowerModel,X)
    img = Image.fromarray(imageData.astype('uint8'))
    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)
    fileName = str(datetime.datetime.now().timestamp()) + ".png" 
    return send_file(file_object,attachment_filename=fileName, mimetype='image/PNG')


@app.route('/generateNumber', methods=['GET'])
def generateNumber():
    X = generate_latent_input(LATENT_DIM)
    # generate images
    imageData =  generateSampleOutput(digitsModel,X)
    img = Image.fromarray(imageData.astype('uint8'),'L')
    # create file-object in memory
    file_object = io.BytesIO()

    # write PNG in file-object
    img.save(file_object, 'PNG')

    # move to beginning of file so `send_file()` it will read from start    
    file_object.seek(0)
    fileName = str(datetime.datetime.now().timestamp()) + ".png" 
    return send_file(file_object,attachment_filename=fileName, mimetype='image/PNG')


@app.route("/")
def entryPage():
    return render_template('index.html')

# run the application
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"),debug=False)



