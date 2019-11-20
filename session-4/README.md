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

So far, we improved our accuracy by using CNN instead of NN, and by augmenting our training set with manufactured examples, this allowed us to improve our accuracy to around 80% instead of 68%. but we can do more. Our challenge is the fact that we have small number of training samples which limit our model abilities of extract good features to predict the correct class. If we look in tensorflow.keras.applications we will find many pre-built [models](https://www.tensorflow.org/api_docs/python/tf/keras/applications "Applications") targeted for image classification. Let's pick one and take a deeper look, we will look at [xception](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception)

###xception model

This model is trained using the [ImageNet](http://www.image-net.org/) dataset which contains 1000 different classes, the training set is roughly 1.2 million image. The Top-1 accuracy of this model on ImageNet is around 79%. 

We can load the pretrained model usig this code

~~~~{.python}
from tensorflow.keras.applications import xception
from tensorflow.keras.utils import plot_model

model = xception.Xception(weights='imagenet',input_shape=targetSize_withdepth)
model.summary()
plot_model(model, "full.png",show_shapes=True)
~~~~

xception.Xception loads our pre-trained model, while the summary method will print summary of the model architecture. This is quite big model, it has around 22M parameters

plot_model function will save an [image](https://github.com/mohmiim/MLIntroduction/tree/master/session-4/images/full.png "full model") of the model architecture

How can this helps us ? this model had been trained to receive images extract features from these images and classify them to be one of 1000 class, we want to train a model to receive an image, extract features from this image and classify it to one of 5 classes. The 2 problems sounds very similar but they have different output. We can benefit from the features the xception model extracted, but we can not befit form its output classes, if we find a way to load this pre-trained model, but ditch the output layer then we have a model that can extract the features from an image but not classify it, we can call that our feature extractor then we create a simple NN that receive these features instead of the raw image and and train it to do our 5 classes classification. This process is called transfer Learning

## 4. Why transfer learning

Transfer learning enable us to benefit from pretrained models, that had been trained on large data sets and apply their feature extraction abilities to our domain. This helps a lot in cases where we have small training set

The following diagram shows the flow of the transfer learning process

<p align="center"> 
<img src="images/transferLearning.png" height="450" >
</p>

## 5. Apply transfer learning to our use-case

The first few steps for loading our data will stay the same as we did in the previous notebook (make sure you have the data augmentation parameters  passed to ImageDataGenerator ) 

The first change is, before we create our model we need to load the exception model without the output layer, and extract the features from our training set. Teh way to do that is using the following code 

~~~~{.python}
from tensorflow.keras.applications import xception

#LOAD THE PRE_TRAINED MODEL
FEATURE_EXTRACTOR = xception.Xception(weights='imagenet',include_top=False,input_shape=targetSize_withdepth)

#Extract the features for our images
features_x = FEATURE_EXTRACTOR.predict_generator(train_generator)
~~~~

Then we create a simple dense model like this 

~~~~{.python}
model = Sequential()
#add our layers
model.add(Flatten(input_shape=features_x.shape[1:]))
model.add(Dense(128,activation=relu))
model.add(Dropout(0.2))
model.add(Dense(128,activation=relu))
model.add(Dropout(0.5))
model.add(Dense(len(y_labels),activation='softmax'))
model.compile(optimizer=Adam(), loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

model.fit(features_x,y_train,epochs=epochs,shuffle=True,verbose=2)
~~~~

the main difference between what we are doing here and what we did before, is that fact that now we are using the method fit to train the model, and passing the input,the features we extracted, and the output, expected classes to the method to start the training.

This [Notebook](https://github.com/mohmiim/MLIntroduction/blob/master/session-4/Session_4_Transfer_Learning.ipynb) has the full code 

Run the tests and check the results 

## 6. Discuss the Result

I found that accuracy ranging from 90% to 93% that is and increase of 10 to 13%. So the conclusion here is that transfer learning really helped us big team, the reason is that the extracted features from the pre-trained model really helped our use case although they had been extracted for a different classification goal. This is a typical finding, when the input is similar (images in this case) even though the classification goals are different the extracted features form one model can be beneficial for the other.


## 7. Assignment

Update the provided [Notebook](https://github.com/mohmiim/MLIntroduction/blob/master/session-4/Session_4_Transfer_Learning.ipynb) to work with other model from tensorflow.keras.applications and compare the results to xception 

