'''
Created on Feb 23, 2020

@author: miim
'''
import numpy as np
from numpy import random, asarray
from matplotlib import pyplot
from tensorflow.keras.models import load_model
from matplotlib import gridspec
import os

LATENT_DIM = 100
LABEL_COUNT = 102
OUTPUT = 'output/images/flower%04d.png'
ROWS = 5
COLUMNS = 50

def generate_latent_input(latentDim, labels_count=80):
    random.seed(30)
    X = random.randn(latentDim * ROWS * COLUMNS)
    X = X.reshape((ROWS*COLUMNS,latentDim))
    X = X.astype('float32')
    X_labels = random.randint(0,labels_count,ROWS*COLUMNS)
    return [X,X_labels]

def saveSamples(samples, epoch):
    fig = pyplot.figure(figsize=(COLUMNS+1, ROWS+1)) 
    gs = gridspec.GridSpec(ROWS, COLUMNS,
                           wspace=0.1, hspace=0.1, 
                           top=1.-0.5/(ROWS+1), bottom=0.5/(ROWS+1), 
                           left=0.5/(COLUMNS+1), right=1-0.5/(COLUMNS+1)) 
    samples = (samples + 1) / 2.0
    for i in range(ROWS):
        for j in range(COLUMNS):
            index = i*COLUMNS + j
            #pyplot.subplot(ROWS, COLUMNS, 1 + index)
            #pyplot.axis('off')
            #pyplot.imshow(samples[index, :, :] )
            ax= pyplot.subplot(gs[i,j])
            ax.imshow(samples[index, :, :])
            ax.axis('off')
            
    # save to file
    filename = OUTPUT % (epoch)
    pyplot.savefig(filename)
    pyplot.close()

def generateSampleOutput(epoch, generator,X, start_label=0):
    X_labels = asarray([x for _ in range(start_label,ROWS) for x in range(start_label,COLUMNS)])
    y = generator.predict([X,X_labels])
    # save plot
    saveSamples(y, epoch)


folder = "Models/"
filenames = [f for f in os.listdir(folder) if  f.endswith('.h5')]
i = 0 

X, _ = generate_latent_input(LATENT_DIM,LABEL_COUNT)

for filename in filenames:
    # load model
    model = load_model(folder + filename)
    # generate images
    generateSampleOutput(i, model,X,0)
    i=i+1
