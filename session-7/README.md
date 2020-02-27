# Introduction to Generative Adversarial Networks


## What will we cover

[1. What is GAN](#1-what-is-gan)

[2. How to design a model to generate images](#2-how-to-design-a-model-to-generate-images)

[3. How to train GAN](#3-how-to-train-gan)

[4. GAN Tips and Tricks](#4-gan-tips-and-tricks)

[5. Generating hand written digits](#5-generating-hand-written-digits)

[6. Generating flower images](#6-generating-flower-images)

[7. Conditional GAN](#7-conditional-gan)




## 1. What is GAN

Generative Adversarial Networks, are machine learning models able to learn from existing images of a specific domain then generate new images based on the features it learned form the training images. GANs follow a zero-sum game in their training where you have 2 models almost fighting againest each other one is the discriminator model which try to learning how to detect the real images form the generated images, and a generator model which try to generate images with enough quality to trick the discriminator into thinking they are real.

It is a cop and thief game, where the thief is trying to confuse the cop and the cop is tryign to catch the thief. For example imagine a discriminator (cop) model, trained to detect images of money, while a generator (thief) is trying to generate fake money with enough quality to confuse the cop. Any time the discriminator detect the fake images as fake that is an indication for the generator that it needs to do better, and  vice versa. 

<p align="center"> 
<img src="images/GvsD.png" height="200">
</p>

**Discriminator:**
The discriminator is typically a simple model architecture, and it will look svery similar to models we built in the previous sessions, it is just a classifier trained to take an image and classify it as real or fake.

<p align="center"> 
<img src="images/discriminator.png" height="400">
</p>

**Generator:**
The generator responsibility is to generate images. It takes input which is a vector of random numbers (called latent space), and generate and image as output. We will get in the details of the Generator in later section

<p align="center"> 
<img src="images/generator.png" height="400">
</p>

## 2. How to design a model to generate images

We have session in the previous sessions how to create classifiers using many different techniques. This covers the discriminator, But the generator is a new kind of a problem we did not face till now. 
Generator job is to receive a vector of random points as input and output a full image. So far when we dealt with images we have been using Conv2D layers to down sample an image to set of features, but here we want to do the opposite where we want to up-sample from a set of features to real image. There are different techniques to achieve that and Keras has a nice layer that we will be using extensively in this session to achieve the goal up sampling this layer is **Conv2DTranspose** this layer will learn how to up-sample during our training process as you will see in the next few sections.

One big difference between, **Conv2DTranspose** and **Conv2D** is the effect of the strides parameter, doe example if you set the strides parameter to (2,2) in **Conv2D** it will basically half the size of your image (an image of 20X20 will become 10X10) while in **Conv2DTranspose** it will double the size of the image (an image of 20X20 will become 40X40).

<p align="center"> 
<img src="images/strides.png" height="200">
</p>

## 3. How to train GAN

As discussed in **[What is GAN](#1-what-is-gan)**, GAN actually consists of 2 models, the discriminator and the generator, and we are trying to train both at the same time and they affect each other, as a result GAN training is a bit more involved than the training we did in previous sessions.

Lets see what happens during one epoch while training GAN, during one epoch we need to update the weights of both the discriminator model and the generator model.  

**Discriminator update** :
  
We want to train a discriminator that will be able to classify images as real or fake, this means during one epoch a set of fake and real images will be passed to the discriminator and it will use these samples to classify them as real or fake meaning it will try to update the weights of the discriminator to produce the correct classifications. 
specific example, lets say we want to create a model able to generate paintings of trees, what we will need is to find a good data set of tree paintings (this will be our real samples), assume our samples size is 400 tree paintings, we need a similar number of fake images of tree paintings, How can we get these?. Well, this is where the generator comes into play, we create a set of 400 inputs for our generator, pass these through the generator, it will generate 400 fake tree paintings (it does not matter that the generator is not trained yet, we will see how it all connect shortly). Now we have 400 real tree paintings, and 400 fake tree painting, we label them as such then call the discriminator model to do one update (fit) using these samples (the 800 images), this will result in updating the discriminator weights to achieve the task of detecting the fake from the real images.

The following figure shows the Discriminator model training process:

<p align="center"> 
<img src="images/DiscriminatorModel.png" height="500">
</p>

**Generator update** :  
 
The Generator goal is a bit different, it is goal is to generate good enough images that the discriminator think they are real. During one epoch we updated the discriminator weights (as per previous section), then we need to update the generator weights too. the input to the generator is a vector or numbers (the latent space), so it is easy to randomly generate set of those, but we wnat the generator to generate realistic images that can confuse the discriminator, to do what we create a model where we stack the generator on top of the discriminator, this means that the output of the generator is passed to the discriminator. If we train this model to classify these fake images as real, it means it will update the weights in a way that make the generator generate real images, but there is a caveat here, the update will try to update both the discriminator and generator weights since they are in the same model, and this means we are messing up the training we did in the previous section, we just want to update the generator model weights in this step, this is easily fixable, we can do that by freezing the discriminator weights in this model, this way it will just update the generator weights which is exactly what we want.  

The following figure shows the GAN model used to train the generator:

<p align="center"> 
<img src="images/GANModel.png" height="500">
</p>

## 4. GAN Tips and Tricks

GAN training as seen in **[How to train GAN](#3-how-to-train-gan)**, is quite involved. And a lot of the training process and parameters adjustment is a trial and error kind of approach. This being said, there are many successfully trained GAN models out there, and people shared their tips and tricks on how to train a GAN
Here few:  

* Use LeakyReLU instead or Relu for the activation of hidden layers with slope of 0.2
* Use gaussian weight initialization for the weights, using RandomNormal from Keras
* Use Adam optimizer with lr = 0.0002 and beta-1=0.5 but if your model suffer from model collapse (it starts to generate the same image from many inputs), lower the learning rate
* Scale your image input to [-1,1] instead of [0,1]
* Add noise to the training, by miss-labeling some samples (like 5%)
* Use label smoothing, so instead of using 1 for real use 0.8 to 1.0 and instead of using 0 for fake use 0 to 0.2
* when training the discriminator, do the training on 2 steps one patch with only fake and one patch with only real
* instead of using pooling layer in the discriminator, use stride of (2,2)
* consider using BatchNormilization layer before the activation (i got mixed result for this one)

These are just few tips and tricks, your results might be different and you should feel comfortable mixing and matching and changing the hyperparameters 

## 5. Generating hand written digits

## 6. Generating flower images

## 7. Conditional GAN
