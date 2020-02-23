'''
Created on Jan 17, 2020

@author: miim
'''
from tensorflow.keras.datasets.mnist import load_data
from matplotlib import pyplot
from skimage.transform import resize
import os.path as path
import numpy as np
from PIL import Image
import scipy.io
import os
from scipy import ndimage
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


def resize_save(samples):
    print("Resizing")
    new_size = (samples.shape[1]*scale_factor, samples.shape[2]*scale_factor)
    resized_training = np.zeros((samples.shape[0], new_size[0], new_size[1]))
    print(number_of_samples)
    i = 0
    for image in X :
        if i==0 :
            print(image.shape) 
        image = resize(image, new_size)
        if i==0 :
            print(image.shape)
        resized_training[i] = image
        if i%1000 == 0 :
            print(i)
        i = i + 1
    np.savez_compressed(FILE_NAME, resized_training)
    return resized_training

def load_samples(location = "data/outline/",width=112, height=112,mode='L', SHOW_SAMPLE=False) :
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
        #array_to_img(data_set[i]).show()
    print(" ")
    if SHOW_SAMPLE : 
        array_to_img(data_set[4]).show()
    print(data_set.shape)
    data_set = data_set.astype('float32')
    return data_set        

def load_labels(data_label = 'labels', file = 'D:/ml/gandatasets/102flowers/imagelabels.mat') :
    print('reading: labels')
    mat = scipy.io.loadmat(file)
    data = mat[data_label]
    data = data.reshape((data.shape[1]))
    data = data-1
    ulabels = sorted(set(data))
    print(ulabels)
    print(data.shape)
    return data,len(ulabels)


'''
currentlabel = 0
for i in range(0,labels.shape[0]):
    if labels[i]!=currentlabel:
        print("Label : " + str(currentlabel) + " changed to : " + str(labels[i]) + " at index : " + str(i))
        currentlabel = labels[i]
'''


def load_real_samples(scale_factor=2, SHOW_SAMPLE=False):
    (X,_),(_,_) = load_data()
    number_of_samples = X.shape[0]
    X = X.astype('float32')
    # scale from [0,255] to [0,1]
    X = (X / 255)
    
    FILE_NAME = "data/resized.npz"
    
    if path.exists(FILE_NAME) :
        resized_training = np.load(FILE_NAME)['arr_0']
    else:
        resized_training = resize_save(X)
    
    
    if SHOW_SAMPLE : 
        
        img = Image.fromarray(X[2]*255)
        img.show()
        img = Image.fromarray(resized_training[2]*255)
        img.show()
    return  np.expand_dims(resized_training, axis=-1)