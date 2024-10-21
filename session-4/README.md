# Session 4: Improve the accuracy

## What will we cover

[1. Accuracy so far and how to improve it](#1-accuracy-so-far-and-how-to-improve-it)

[2. Apply samples manufacturing to our use-case](#2-apply-samples-manufacturing-to-our-use-case)

[3. Transfer learning](#3-transfer-learning)

[4. Why transfer learning](#4-why-transfer-learning)

[5. Apply transfer learning to our use-case](#5-apply-transfer-learning-to-our-use-case)

[6. Discuss the Result](#6-discuss-the-result)

[7. Assignment](#7-assignment)

## 1. Accuracy so far and how to improve it

So far, we have managed to get accuracy in the range of 70% (+/- 3), which is OK but not great. It is a good time to think about why we lack accuracy.  It is All about data; our training sample is too small compared to the problem we are trying to tackle. Oh, OK, then, let's have more samples; this is easier said than done. Deep learning has a hunger for data and getting clean, labelled samples is a time and money-consuming task.

What if there is a way to manufacture samples automatically? Yes, there is. In session 2, we used the method [ImageDataGenerator](https://github.com/mohmiim/MLIntroduction/tree/master/session-2#2-loading-images-training-set-using-tensorflow "Image data Generator") to load our training samples, but this method can do more. It can do Data augmentation based on the parameters we use.

ImageDataGenerator supports different types of data augmentation options:
 - width_shift_range and height_shift_range shift the image along with width or height.
 - horizontal_flip and vertical_flip flips image.
 - rotation_range, Image rotations
 - brightness_range, Image brightness.
 - zoom_range, Image zoom.

## 2. Apply samples manufacturing to our use-case

Let's continue with our code from session 3 and modify the ImageDataGenerator code to look like this. 

~~~~{.python}

train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          zoom_range=0.2,
                                          fill_mode='nearest')
~~~~

Then, let's re-train our model and run the tests; what do you find?

This [Notebook](https://github.com/mohmiim/MLIntroduction/blob/master/session-4/Session_4_Data_Augmentationt.ipynb "Training Augmentation") has the full code; you will notice that the accuracy increased to around 80%, this is a big jump keeping in mind that this 8-10% increase came with minimum effort and using the same CNN

That is cool, but can we do better? 

## 3. Transfer learning

So far, we have improved our accuracy by using CNN instead of NN, and by augmenting our training set with manufactured examples, this allowed us to improve our accuracy to around 80% instead of 68%. But we can do more. Our challenge is that we have few training samples, which limits our model's ability to extract good features to predict the correct class. If we look in tensorflow.keras.applications we will find many pre-built [models](https://www.tensorflow.org/api_docs/python/tf/keras/applications "Applications") targeted for image classification. Let's pick one and take a deeper look; we will look at [xception](https://www.tensorflow.org/api_docs/python/tf/keras/applications/xception)

###xception model

This model is trained using the [ImageNet](http://www.image-net.org/) dataset, which contains 1000 different classes. The training set is roughly 1.2 million images. This model's Top-1 accuracy on ImageNet is around 79%. 

We can load the pre-trained model using this code.

~~~~{.python}
from tensorflow.keras.applications import xception
from tensorflow.keras.utils import plot_model

model = xception.Xception(weights='imagenet',input_shape=targetSize_withdepth)
model.summary()
plot_model(model, "full.png",show_shapes=True)
~~~~

xception.Xception loads our pre-trained model, while the summary method prints a summary of the model architecture. This is quite a big model; it has around 22M parameters.

plot_model function will save an [image](https://github.com/mohmiim/MLIntroduction/tree/master/session-4/images/full.png "full model") of the model architecture

Do you know how this can help us? This model had been trained to receive images, extract features from these images, and classify them as one of 1000 classes; we want to train a model to receive an image, extract features from it, and classify it into one of 5 classes. The two problems sound very similar, but they have different outputs. We can benefit from the features the xception model extracted, but we can not befit from its output classes; if we find a way to load this pre-trained model but ditch the output layer, then we have a model that can extract the features from an image but not classify it, we can call that our feature extractor then we create a simple NN that receive these features instead of the raw image and train it to do our five classes classification. This process is called transfer Learning.

## 4. Why transfer learning

Transfer learning enables us to benefit from pre-trained models trained on large data sets and apply their feature extraction abilities to our domain. This helps a lot in cases where we have a small training set

The following diagram shows the flow of the transfer learning process.

<p align="center"> 
<img src="images/transferLearning.png" height="450" >
</p>

## 5. Apply transfer learning to our use-case

The first few steps for loading our data will stay the same as we did in the previous notebook (make sure you have the data augmentation parameters  passed to ImageDataGenerator ) 

The first change is that before we create our model, we need to load the exception model without the output layer and extract the features from our training set. The way to do that is by using the following code. 

~~~~{.python}
from tensorflow.keras.applications import xception

#LOAD THE PRE_TRAINED MODEL
FEATURE_EXTRACTOR = xception.Xception(weights='imagenet',include_top=False,input_shape=targetSize_withdepth)

#Extract the features for our images
features_x = FEATURE_EXTRACTOR.predict_generator(train_generator)
~~~~

Then we create a simple dense model like this. 

~~~~{.python}
model = Sequential()
#add our layers
model.add(Flatten(input_shape=features_x.shape[1:]))
model.add(Dense(128,activation=relu))
model.add(Dropout(0.1))
model.add(Dense(64,activation=relu))
model.add(Dense(len(y_labels),activation='softmax'))
history = model.compile(optimizer=Adam(lr=0.0001), loss="categorical_crossentropy", metrics=['accuracy'])
model.summary()

model.fit(features_x,y_train,epochs=epochs,shuffle=True,verbose=2)
~~~~

The main difference between what we are doing here and what we did before is that now we are using the method fit to train the model and passing the input, the features we extracted, and the output, expected classes to the method to start the training.

This [Notebook](https://github.com/mohmiim/MLIntroduction/blob/master/session-4/Session_4_Transfer_Learning.ipynb) has the full code 

Run the tests and check the results. 

## 6. Discuss the Result

I found that accuracy ranging from 90% to 93% that is an increase of 10 to 13%. So the conclusion here is that transfer learning really helped us big time; the reason is that the extracted features from the pre-trained model helped our use case, although they had been extracted for a different classification goal. This is a typical finding when the input is similar (images in this case) even though the classification goals are different the extracted features form one model can be beneficial for the other.


## 7. Assignment

Update the provided [Notebook](https://github.com/mohmiim/MLIntroduction/blob/master/session-4/Session_4_Transfer_Learning.ipynb) to work with another model from tensorflow.keras.applications and compare the results to xception 

