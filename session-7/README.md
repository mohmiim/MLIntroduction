# Introduction to Generative Adversarial Networks


## What will we cover

[1. What is GAN](#1-what-is-gan)

[2. How to design a model to generate images](#2-how-to-design-a-model-to-generate-images)

[3. How to train GAN](#3-how-to-train-gan)

[4. GAN best practice](#4-gan-best-practice)

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

## 3. How to train GAN]

## 4. GAN best practice

## 5. Generating hand written digits

## 6. Generating flower images

## 7. Conditional GAN
