'''
Created on Feb 23, 2020

@author: miim
'''

from numpy import expand_dims
from keras.datasets.mnist import load_data

from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
import os

def load_samples(location = "data/outline/",width=112, height=112,mode='L') :
    folder = location
    size = (width,height)
    onlyfiles = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]
    data_set = np.zeros((len(onlyfiles), size[0], size[1], 3))
    print("Working with {0} images".format(len(onlyfiles)))
    print("Image examples: ")
    i=0
    for _file in onlyfiles:
        if i%100 == 0:
            print('.',end="")
        img = Image.open(folder + _file).convert(mode) 
        img = img.resize(size)
        x = img_to_array(img)  
        data_set[i] = x
        i = i +1
    print(data_set.shape)
    data_set = data_set.astype('float32')
    data_set = (data_set - 127.5) / 127.5
    return data_set  

from numpy.random import randn

LATENT_DIM = 100

def generate_latent_input(latentDim, count):
    X = randn(latentDim * count)
    X = X.reshape((count,latentDim))
    X = X.astype('float32')
    return X

from numpy import zeros
def create_generated_samples(generator, latentDim, count):
    X = generate_latent_input(latentDim, count)
    gen_images = generator.predict(X)
    # labels here will be fake ==> 0
    y = zeros((count,1))
    return gen_images, y



from numpy.random import randint
from numpy import ones

def generate_real_samples(dataset, n_samples):
    index = randint(0, dataset.shape[0], n_samples)
    X = dataset[index]
    # mark them as real
    y = ones((n_samples, 1))
    return X, y


from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.initializers import RandomNormal 

DISC_FILTER_SIZE = 5
DISC_LEAKY_ALPHA = 0.2 

init = RandomNormal(stddev=0.02)
def createDiscConvLayer(model):
    model.add(Conv2D(128, (DISC_FILTER_SIZE,DISC_FILTER_SIZE),
                     strides=(2, 2),
                     padding='same',
                     kernel_initializer=init))
    model.add(LeakyReLU(alpha=DISC_LEAKY_ALPHA))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout,Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

INPUT_SIZE = (80,80,3)
DISC_DROPOUT = 0.4
def create_discriminator(input_shape=INPUT_SIZE):
    print("Creating Discriminator")
    model = Sequential()
    model.add(Conv2D(40, (DISC_FILTER_SIZE,DISC_FILTER_SIZE),
                      padding='same',
                      kernel_initializer=init,
                      input_shape=input_shape))
    model.add(LeakyReLU(alpha=DISC_LEAKY_ALPHA))
    # down sample to 40 X 40
    createDiscConvLayer(model)
    # down sample to 20 X 20
    createDiscConvLayer(model)
    # down sample to 10 X 10
    createDiscConvLayer(model)
    # down sample to 5 X 5
    createDiscConvLayer(model)

    model.add(Flatten())
    model.add(Dropout(DISC_DROPOUT))
    activation = 'sigmoid'
    loss= 'binary_crossentropy'
    model.add(Dense(1, activation=activation))
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    print("Created Discriminator")
    model.summary()
    return model

from tensorflow.keras.layers import Conv2DTranspose

GEN_FILTER_SIZE = 5
GEN_LEAKY_ALPHA = 0.2

def addGenConvTransPoseLayer(model):
  model.add(Conv2DTranspose(128, (GEN_FILTER_SIZE,GEN_FILTER_SIZE),
                            strides=(2,2),
                            padding='same'))
  model.add(LeakyReLU(alpha=GEN_LEAKY_ALPHA))
  
from tensorflow.keras.layers import Reshape

def create_generator(latent_dim = LATENT_DIM):
    print("Creating Genertor")
    model = Sequential()
    # foundation for 64x64 image
    n_nodes = 128 * 5 * 5
    model.add(Dense(n_nodes, input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((5, 5, 128)))
    # up sacle to 10 * 10
    addGenConvTransPoseLayer(model)
    # upsample to 20*20
    addGenConvTransPoseLayer(model)
    # upsample to 40*40
    addGenConvTransPoseLayer(model)
    # upsample to 80*80
    addGenConvTransPoseLayer(model)
    # output layer
    model.add(Conv2D(3, (5,5),
                     activation='tanh',
                     padding='same',
                     kernel_initializer=init))
    print("Created Generator")
    model.summary()
    return model

def create_gan(generator, discriminator):
    print("Creating GAN")
    # freeze the weights of the discriminator
    discriminator.trainable = False
    # connect them
    model = Sequential()
    # add generator
    model.add(generator)
    # add the discriminator
    model.add(discriminator)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    loss= 'binary_crossentropy'
    model.compile(loss=loss, optimizer=opt)
    print("Created GAN")
    model.summary()
    return model

from matplotlib import pyplot

def saveSamples(samples, epoch, n=5):
    samples = (samples + 1) / 2.0
    for i in range(n * n):
        pyplot.subplot(n, n, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(samples[i, :, :] )
            
    # save to file
    filename = 'generated_plot_e%04d.png' % (epoch+1)
    pyplot.savefig(filename)
    pyplot.close()

def generateSampleOutput(epoch, generator, n_samples=10):
    X = generate_latent_input(LATENT_DIM, n_samples*n_samples)
    y = generator.predict(X)
    # save plot
    saveSamples(y, epoch, n_samples)
    # save the generator model tile file
    filename = 'generator_model_%04d.h5' % (epoch + 1)
    generator.save(filename)

import math

def train(generator, discriminator, gan, dataset, latent_dim, n_epochs=500, n_batch=256):
  batches_count = math.ceil(dataset.shape[0] / n_batch)
  half_batch = int(n_batch / 2)
  # manually enumerate epochs
  for i in range(n_epochs):
    # enumerate batches over the training set
    print(" ")
    print('>Epoch:%d' % (i+1), end = " ")
    for j in range(batches_count):
      # get randomly selected 'real' samples
      X_real, y_real = generate_real_samples(dataset, n_batch)
      # generate 'fake' examples
      X_fake, y_fake = create_generated_samples(generator, latent_dim, half_batch)
      # update discriminator model weights
      lossReal, _ = discriminator.train_on_batch(X_real, y_real)
      lossFake, _ = discriminator.train_on_batch(X_fake, y_fake)
      # prepare points in latent space as input for the generator
      X_gan = generate_latent_input(latent_dim, n_batch)
      # mark fake as real
      y_gan = ones((n_batch, 1))
      # update the generator via the discriminator's error
      loss_generator = gan.train_on_batch(X_gan, y_gan)
      print(".", end="")
      # summarize loss on this batch
    if i%10 == 0 :
      generateSampleOutput(i+1,generator,4)

dataset  = load_samples("data/102flowers/jpg/", width=80, height=80, mode="RGB")

# create the discriminator
discriminator = create_discriminator()
# create the generator
generator = create_generator(LATENT_DIM)
# create the gan
gan = create_gan(generator, discriminator)

# train model
train(generator, discriminator, gan, dataset, LATENT_DIM)




