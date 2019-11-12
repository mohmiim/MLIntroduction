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

But Why is that important to understand, The fully  connected nature of the Dense layer impacts how many parameters we need to train during the training phase, and it impacts how long training will take and how big it the model. For example if we have neural network that accepts 5 inputs and have 3 hidden layers 10 nodes each and one output layer of 2 nodes. The number of parameters in this networks will be N = 5 * 10 + 10 * 10 + 10 * 10 + 10 * 2 = 270 (this calculation does not consider the extra B node)

Can we reduce this, so we can create very complex netwroks without having 100s of millions of parameters to train ?!! Welcome to convolution neural networks or CNN


## 2. What is Convolutional Neural Network





## 3. Why Convolutional Neural Network




## 4. Using Convolutional Neural Network for chart recognition



## 5. Compare accuracy of the CNN model to the NN model


## 6. Assignment

