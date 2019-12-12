'''
Created on Nov. 26, 2019

@author: mohammedmostaqfa
'''
import numpy as np
from tensorflow.keras.applications import xception
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.utils import plot_model
from matplotlib import pyplot as plt

def plotModel(model, imagePath="model.png", show_shapes=True,show_layer_names=False,expand_nested=False,dpi=96):
    plot_model(model,
                to_file=imagePath,
                show_shapes=show_shapes,
                show_layer_names=show_layer_names,
                expand_nested=expand_nested,
                dpi=dpi)


y_train = []
y_labels = {}

small_data_folder = "../data/"


#default sizes
Image_Width = 100
Image_Height = 100
Image_Depth = 3
targetSize = (Image_Width,Image_Height)
targetSize_withdepth = (Image_Width,Image_Height,Image_Depth)

image_data = image.load_img("../images/10.png",target_size=(100,100))
image_array = image.img_to_array(image_data)
x_train = []
x_train.append(image_array)
x_train = np.array(x_train)
x_train = xception.preprocess_input(x_train)

LAYERS_COUNT = 3

# NOW WE LOAD THE PRE_TRAINED MODEL
model = load_model("../model/CNNDetector_5_enr.h5")
layer_outputs = [layer.output for layer in model.layers[:LAYERS_COUNT]] 
# Extracts the outputs of the top 12 layers
layersModel = Model(inputs=model.input, outputs=layer_outputs)
out = layersModel.predict(x_train) 
plotModel(layersModel,"image.png")

layer_names = []
for layer in layersModel.layers[1:LAYERS_COUNT+1]:
    layer_names.append(layer.name) 
    
images_per_row = 16

for layer_name, layer_activation in zip(layer_names, out):
    if not "max_pool" in layer_name :
        print (layer_activation.shape)
        n_features = layer_activation.shape[-1] # Number of features in the feature map
        size = layer_activation.shape[1] #The feature map has shape (1, size, size, n_features).
        n_cols = n_features // images_per_row # Tiles the activation channels in this matrix
        display_grid = np.zeros((size * n_cols, images_per_row * size))
        for col in range(n_cols): # Tiles each filter into a big horizontal grid
            for row in range(images_per_row):
                index = col * images_per_row + row
                channel_image = layer_activation[0,
                                                 :, :,
                                                 index]
                #channel_image -= channel_image.mean() # Post-processes the feature to make it visually palatable
                #channel_image /= channel_image.std()
                #channel_image *= 64
                #channel_image += 128
                #channel_image = np.clip(channel_image, 0, 255).astype('uint8')
                display_grid[col * size : (col + 1) * size, # Displays the grid
                             row * size : (row + 1) * size] = channel_image
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1],
                            scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='gray')

plt.show()
