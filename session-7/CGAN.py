'''
Created on Jan 17, 2020

@author: miim
'''
import numpy as np
from numpy.random import randn,randint
from numpy import ones,zeros
from numpy import asarray

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate,Dropout,LeakyReLU,Reshape,Flatten, BatchNormalization
from tensorflow.keras.layers import Input,Dense,Embedding,Conv2D,Conv2DTranspose
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import RandomNormal 
from tensorflow import ones_like, zeros_like
import tensorflow as tf

from DinoGen.ImageUtils import load_samples,load_labels
from matplotlib import pyplot
import math

LATENT_DIM = 100
INPUT_SIZE = (80,80,3)
GEN_FILTER_SIZE = 3
GEN_LEAKY_ALPHA = 0.2
PATCH_SIZE = 128
RGB = True
EMBEDDING_COUNT = 400

USE_MSE = False

def get_patch_samples(dataset, patchindex, patchSize):
    images, labels = dataset
    start = (patchindex - 1) * patchSize
    end = patchindex * patchSize  - 1
    if end > images.shape[0] :
        end = images.shape[0]
    X = images[start:end]
    X_labels = labels[start:end]
    # real label ==> 1
    y = ones((X.shape[0],1))
    return [X, X_labels], y

# select real samples
def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # retrieve selected images
    X = images[ix]
    X_labels = labels[ix]
    # generate 'real' class labels (1)
    y = ones((n_samples, 1))
    return [X,labels], y

def generate_latent_input(latentDim, count, labels_count=80):
    X = randn(latentDim * count)
    X = X.reshape((count,latentDim))
    X = X.astype('float32')
    X_labels = randint(0,labels_count,count)
    return [X,X_labels]

def create_generated_samples(generator, latentDim, count,labels_count):
    X, X_labels = generate_latent_input(latentDim, count,labels_count)
    gen_images = generator.predict([X,X_labels])
    # labels here will be fake ==> 0
    y = zeros((count,1))
    return [gen_images,X_labels], y

DISC_DROPOUT = 0.4
DISC_FILTER_SIZE = 5
DISC_LEAKY_ALPHA = 0.2 

init = RandomNormal(stddev=0.02)
def createDiscConvLayer(prev):
    newLayer = Conv2D(128,
                      (DISC_FILTER_SIZE,DISC_FILTER_SIZE),
                      strides=(2, 2),
                      padding='same',
                      kernel_initializer=init)(prev)
    newLayer = LeakyReLU(alpha=DISC_LEAKY_ALPHA)(newLayer)
    return newLayer
    
# the discriminator 
def create_discriminator(input_shape=INPUT_SIZE, nLabels=80):
    print("Creating Discriminator")
    # we have 2 inputs, one for image and one for labels
    in_labels = Input(shape=(1))
    in_image = Input(shape=input_shape)
    # now we need to encode the label and shape like an extra channel
    label = Embedding(nLabels,EMBEDDING_COUNT )(in_labels)
    nodes_count = input_shape[0] * input_shape[1]
    label = Dense(nodes_count)(label)
    label = Reshape((input_shape[0],input_shape[1],1))(label)
    # add the encoded label as a channel to the image input
    image_and_label = Concatenate()([in_image,label])
    
    # the rest is our typical discrimonator
    feature_extractor = Conv2D(40,
                               (DISC_FILTER_SIZE,DISC_FILTER_SIZE),
                               padding='same',
                               kernel_initializer=init)(image_and_label)                    
    feature_extractor = LeakyReLU(alpha=DISC_LEAKY_ALPHA)(feature_extractor)
    # down sample to 80 X 80
    #feature_extractor = createDiscConvLayer(feature_extractor)
    # down sample to 40 X 40
    feature_extractor = createDiscConvLayer(feature_extractor)
    # down sample to 20 X 20
    feature_extractor = createDiscConvLayer(feature_extractor)
    # down sample to 10 X 10
    feature_extractor = createDiscConvLayer(feature_extractor)
    # down sample to 5 X 5
    feature_extractor = createDiscConvLayer(feature_extractor)

    feature_extractor = Flatten()(feature_extractor)
    feature_extractor = Dropout(DISC_DROPOUT)(feature_extractor)
    
    activation = 'sigmoid'
    loss= 'binary_crossentropy'
    if USE_MSE : 
        activation = "linear"
        loss = 'mse'
        
    output_layer = Dense(1, activation=activation)(feature_extractor)
    
    # create the model
    model = Model([in_image,in_labels], output_layer)
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    print("Created Discriminator")
    model.summary()
    plot_model(model,to_file="discriminator.png", show_shapes=True)
    return model

def addGenConvTransPoseLayer(prev):
    newLayer =Conv2DTranspose(128,
                              (4,4),
                              strides=(2,2),
                              padding='same',
                              #use_bias = False,
                              kernel_initializer=init)(prev)
    #newLayer = BatchNormalization()(newLayer)
    newLayer = LeakyReLU(alpha=0.2)(newLayer)
    return newLayer
    

def create_generator(latent_dim = LATENT_DIM,nLabels=80):
    BASE_IMAGE_DIM = 5;
    print("Creating Genertor")
    #we have 2 inputs,the latent space vector, and the label
    input_label = Input(shape=(1))
    input_lat = Input(shape=(latent_dim))
    #encode the label to be like extra channel for latent space
    label = Embedding(nLabels,int(latent_dim/2))(input_label)
    label = Dense(BASE_IMAGE_DIM*BASE_IMAGE_DIM)(label)
    label = Reshape((BASE_IMAGE_DIM,BASE_IMAGE_DIM,1))(label)
    
    # base for BASE_IMAGE_DIM*BASE_IMAGE_DIM image
    n_nodes = 128 * BASE_IMAGE_DIM * BASE_IMAGE_DIM
    lat_space = Dense(n_nodes)(input_lat)
    lat_space = LeakyReLU(alpha=0.2)(lat_space)
    lat_space = Reshape((BASE_IMAGE_DIM, BASE_IMAGE_DIM, 128))(lat_space)
    #add the label as another filter output
    generator = Concatenate()([lat_space,label])
    
    # up sacle to 10 * 10
    generator = addGenConvTransPoseLayer(generator)
    # upsample to 20*20
    generator = addGenConvTransPoseLayer(generator)
    # upsample to 40*40
    generator = addGenConvTransPoseLayer(generator)
    # upsample to 80*80
    generator = addGenConvTransPoseLayer(generator)
    # upsample to 160*160
    #generator = addGenConvTransPoseLayer(model)
    # output layer
    output_layer = Conv2D(3,
                          (5,5),
                          activation='tanh',
                          padding='same',
                          #use_bias = False,
                          kernel_initializer=init)(generator)

    #create the model
    model = Model([input_lat,input_label], output_layer)
    print("Created Generator")
    model.summary()
    plot_model(model,to_file="generator.png", show_shapes=True)
    return model

def create_gan(generator, discriminator):
    print("Creating GAN")
    # make weights in the discriminator not trainable
    discriminator.trainable = False
    # extract the latent space and the label
    latent_space , label = generator.input
    generator_output_image = generator.output
    
    # connect the generator output image and the label to the discriminator 
    gan_output = discriminator([generator_output_image,label])
    
    #create the model
    model = Model([latent_space,label],gan_output)
    
    # compile model
    opt = Adam(lr=0.0002, beta_1=0.5)
    loss= 'binary_crossentropy'
    if USE_MSE : 
        loss = 'mse'
    model.compile(loss=loss, optimizer=opt)
    print("Created GAN")
    model.summary()
    plot_model(model,to_file="CGAN.png", show_shapes=True)
    return model


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

def generateSampleOutput(epoch, generator, labels_count, n_samples=10):
    X, X_labels = generate_latent_input(LATENT_DIM, n_samples*n_samples,labels_count)
    X_labels = asarray([x for _ in range(1,n_samples+1) for x in range(1,n_samples+1)])
    y = generator.predict([X,X_labels])
    # save plot
    saveSamples(y, epoch, n_samples)
    # save the generator model tile file
    filename = 'generator_model_%04d.h5' % (epoch + 1)
    generator.save(filename)



# train the generator and discriminator
def train(generator, discriminator, cgan, dataset, latent_dim,labels_count,n_epochs=12000, n_batch=128):
    batches_count = math.ceil(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        # for all batches
        for j in range(batches_count):
            # get random  samples
            #[X_real,X_real_labels], y_real = generate_real_samples(dataset, half_batch)
            [X_real,X_real_labels], y_real = get_patch_samples(dataset, j+1, n_batch)
            # generate 'fake' examples
            [X_fake,X_fake_labels], y_fake = create_generated_samples(generator, latent_dim, half_batch,labels_count)
            # update discriminator model weights
            dLossReal, _ = discriminator.train_on_batch([X_real,X_real_labels], y_real)
            dLossFake, _ = discriminator.train_on_batch([X_fake,X_fake_labels], y_fake)
            # prepare points in latent space as input for the generator
            [X_gan,X_gan_labels] = generate_latent_input(latent_dim, n_batch,labels_count)
            # create inverted labels for the fake samples
            y_gan = ones((n_batch, 1))
            # update the generator via the discriminator's error
            g_loss = cgan.train_on_batch( [X_gan,X_gan_labels], y_gan)
            # summarize loss on this batch
            print('>Epoch:%d, Patch:%d/%d, dReal=%.3f, dFake=%.f g=%.3f' % (i+1, j+1, batches_count, dLossReal,dLossFake, g_loss))
        if i%10 == 0 :
            print("saving")
            generateSampleOutput(i+1,generator,labels_count,4)

dataset  = load_samples("data/102flowers/jpg/", width=80, height=80, mode="RGB")
labels, count = load_labels(file="data/102flowers/imagelabels.mat")

# lets scale data set to  -1 to 1
dataset  = (dataset - 127.5 ) / 127.5

X_train = [dataset,labels]


# create the discriminator
discriminator = create_discriminator(INPUT_SIZE,count)

# create the generator
generator = create_generator(LATENT_DIM,count)

# create the gan
gan = create_gan(generator, discriminator)


generateSampleOutput(0,generator,count,4)

# train model
train(generator, discriminator, gan, X_train, LATENT_DIM,count)

