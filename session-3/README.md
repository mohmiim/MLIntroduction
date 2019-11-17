# Session 3: Introduction to Convolutional Neural Network (CNN)


## What will we cover

[1. What is Dense Layer](#1-what-is-dense-layer)

[2. What is Convolutional Neural Network](#2-what-is-convolutional-neural-network)

[3. Why Convolutional Neural Network](#3-why-convolutional-neural-network)

[4. Using Convolutional Neural Network for chart recognition](#4-using-convolutional-neural-network-for-chart-recognition)

[5. Compare accuracy of the CNN model to the NN model](#5-compare-accuracy-of-the-cnn-model-to-the-nn-model)

[6. Assignment](#6-assignment)

## 1. What is Dense Layer

We went through how single neuron and multi neuron (multi layers as well) neural networks work, one observation you would have  seen is that layers in neural networks are fully connected. Meaning that every node in one layer is connected to all nodes in the layers before and after it  (Hence the layer type we used in tensorflow is called Dense). 

<p align="center"> 
<img src="images/dense.png" >
</p>

But Why is that important to understand, The fully  connected nature of the Dense layer impacts how many parameters we need to train during the training phase, and it impacts how long training will take and how big it the model. For example if we have neural network that accepts 5 inputs and have 3 hidden layers 10 nodes each and one output layer of 2 nodes. The number of parameters in this networks will be N = 5 \* 10 + 10 \* 10 + 10 \* 10 + 10 \* 2 = 270 (this calculation does not consider the extra B node)

Can we reduce this, so we can create very complex netwroks without having 100s of millions of parameters to train ?!! Welcome to convolution neural networks or CNN


## 2. What is Convolutional Neural Network

Convolution Neural Network is a special kind of neural network that had been proven to work very well with images. For example recognizing faces, animals, different types of objects and so on.  

To understand how CNN works we need to first understand what is the Convolution operation. Convolution operation is applying a specific filter to the image to extract specific feature by considering small quares of the inpout data and maintain their relationship. Let's assume that our image is I and the Filter is K how can we apply the convolution operation  ?

<p align="center"> 
<img src="images/conv.png" >
</p>

In the figure above, convolution is I\*K the filter K is moved across I one step at a time and for each step that part of I is multiplied by K we keep doign that till the filter had been moved over all pixels in I and we get the output image I*K

You will notice that the output did shrink in width and height, in the example above the input was 7\*7 but the output was 5\*5. There are multiple or parameters that affects the convolution operation, in the previous example we the filter moved one pixel at a time, this is called the stride which you can control to be suitable for your use case. Another factor is do you do padding or not, in the example above the output image shrank because we did not do padding so the corner pixels could nto be convoluted, while we could have added a rectangle of 0's around the image this would have increased the input size to 9\*9 so the output would have been 7\*7

<p align="center"> 
<img src="images/padding.png" height="350" width="370" >
</p>

The following image will show the the effect of applying a specific convolution filter on an image 

<p align="center"> 
<img src="images/convSample.png" height="310" width="900" >
</p>

This [notebook](https://github.com/mohmiim/MLIntroduction/blob/master/session-3/ImageConv.ipynb "conv sample") contains code showing the effect of the filter on the image, so you can experiment with different values and different images. Some filter values can extract horizontal edges, some extract vertical and so on. 

When we train a convolution Neural Network it try to find the best filters that extract the best features to identify the different classes we are training it to recognize.



## 3. Why Convolutional Neural Network




## 4. Using Convolutional Neural Network for chart recognition



## 5. Compare accuracy of the CNN model to the NN model


## 6. Assignment

