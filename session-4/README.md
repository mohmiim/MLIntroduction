# Session 4 : Improve the accuracy

## What will we cover

[1. Accuracy so far and how to improve it](#1-accuracy-so-far-and-how-to-improve-it)

[2. Apply samples manufacturing to our use-case](#2-apply-samples-manufacturing-to-our-use-case)

[3. Transfer learning](#3-transfer-learning)

[4. Why transfer learning](#4-why-transfer-learning)

[5. Apply transfer learning to our use-case](#5-apply-transfer-learning-to-our-use-case)

[6. Discuss the Result](#6-discuss-the-result)

[7. Assignment](#7-assignment)

## 1. Accuracy so far and how to improve it

So far we managed to get accuracy in the range of 70% (+/- 3), which is OK but not great. It is a good time to think why is that why we do not have better accuracy?  It is All about data, our training sample is too small compared to the problem we are trying to tackle. Oh, OK then lets have more samples, this is easier said than done, deep learning has hunger to data and getting clean labeled samples is time and money consuming task.

What if there is a way to automatically manufacture samples automatically ? Yes there is, in session 2 we used the method [ImageDataGenerator](https://github.com/mohmiim/MLIntroduction/tree/master/session-2#2-loading-images-training-set-using-tensorflow "Image data Generator") to load our training samples, but this method can do more, it can do Data augmentation based on the parameters we use.

ImageDataGenerator support different types of data augmentation options:
 - width_shift_range and height_shift_range shifts image along width or height.
 - horizontal_flip and vertical_flip flips image.
 - rotation_range, Image rotations
 - brightness_range, Image brightness.
 - zoom_range, Image zoom.

## 2. Apply samples manufacturing to our use-case

Lets continue with our code from session 3, modify the ImageDataGenerator code to look like this 

~~~~{.python}

train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          zoom_range=0.2,
                                          fill_mode='nearest')
~~~~

Then lets re-train our model and run the tests, what do you find ?

This [Notebook](https://github.com/mohmiim/MLIntroduction/blob/master/session-4/Session_4_Data_Augmentationt.ipynb "Training Augmentation") has the full code, you will notice that the accuracy increased to around 80%, this is a big jump keeping in mind that this 8-10% increase came with minimum effort and using the same CNN

That is cool, but can we do better ? 

## 3. Transfer learning

## 4. Why transfer learning

## 5. Apply transfer learning to our use-case

## 6. Discuss the Result

## 7. Assignment