# Session 2: Recognizing chart type from an image

## What will we cover

[1. What are we going to build](#1-what-are-we-going-to-build)

[2. Loading images training set using tensorflow](#2-loading-images-training-set-using-tensorflow)

[3. Designing a Neural network model for chart recognition](#3-designing-a-neural-network-model-for-chart-recognition)

[4. Testing model performance](#4-testing-model-performance)

[5. Confusion matrix](#5-confusion-matrix)

[6. Saving the Model](#6-saving-the-model)

[7. Assignment](#7-assignment)

[8. Observations and Conclusions](#8-observations-and-conclusions)  

## 1. What are we going to build

   In session 1, we saw how to approximate linear and quadratic functions using Neural networks. This was a simple enough project to get introduced to Tensorflow and how to use it to build Neural Network models. Now it is time to move to a more fun project, a project that will be very hard (if not impossible to do with classic programming). Let's build a system when given an image of a chart it can tell us what type of chart is in the image for example, If the system receives this image: 

<p align="center"> 
<img src="images/bar.png" height="350" width="650">
</p>

   It will say this is a bar_chart 
   But if it receive, this image
<p align="center"> 
<img src="images/pie.jpg" height="250" width="250">
</p>  
   
   It will say, this is a pie_chart 
   
   First, we need to think about is what will be the input to our model? In all the cases we have seen so far the input was quite simple it was either 1 or 2 simple numbers, but in this case, what would it be?
	We can not just feed the image as an image to the model for training, we need to represent it as numbers. The easiest way is to feed in the pixel values as the input to the model. Basically, if we have a true-color image we can represent every pixel using 3 integer values to represent the RGB value of the pixel. In other words, if�our image is W width and H height in pixels, we represent it as W*H*3 numbers. This way we can feed it to the Neural network and start the training process.
   
   The following diagram shows how this will look for an image that is 28 X 28 pixels

<p align="center"> 
<img src="images/pixels.png" height="400" width="600">
</p>    
   
## 2. Loading images training set using TensorFlow
   In the last section, we discussed how to present images as input to a neural network, let's talk a bit about training sets and testing sets.
   What we have been doing so far, is called supervised learning, which is training a model by giving it a set of inputs and the expected outputs. Then the model can learn from these input/outputs the rules we need to produce the correct output, These inputs/outputs are called the training set because it had been used to train the model. But how can we check how good is our model?
   
   If we use the same inputs we used for the training to validate it, this will be a misleading measure of the quality of the model, since the model have seen these inputs already and knows what should be the output, this can be used only to measure the training accuracy. But in real use-cases, the model will receive inputs it did not see before. This brings us to the testing data set, which is a set of inputs and their outputs that the model did not see during training. we use this testing dataset to measure the model accuracy to see if it is really able to recognize input it did not see before.
   
   OK, so we now know we need a training set and testing test. Let's see how we prepare the folders containing our images. We start by creating this folder structure
   
<img src="images/smallData.png" height="150" width="150">

   We created separate folders for the training vs testing, and in each folder, we created subfolders one for each type of chart we are going to recognize. For example, in this case, we have samples for bar and pie charts.

Next step is to load the images in these folders and prepare to be fed into our model, Tensorflow has a great utility called ``ImageDataGenerator``, we can use it to load our data the following code shows how this can be done 

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input

train_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
train_generator = train_data_generator.flow_from_directory("../smallData/train",
                                                           batch_size=32,
                                                           target_size=(100,100))
```

What did we do in this code? first step we created the ImageDataGenerator, when you create an ImageDataGenerator you can pass it parameters to control how it works, for example, you can pass it a rescale factor that will be multiplied by all pixel values (to make all values range between 0 and 1), you can also pass it a pre-processing function that will be called on all inputs this can be your own function or one of the tensorflow.keras functions, i found that using `` tensorflow.keras.applications.mobilenet.preprocess_input `` function usually produce good results for me, that is why you will find me passing it in the code when I create the image generator.

Then we called the method flow_from_directory to create our train data generator, this method receives the folder to load the images from, target size it will scale the images to fit (in this case we used 100 X 100), and the batch size, the batch size is basically the number of images to be yielded from the generator per batch.


* Dealing with files in google colab

Since we are starting to deal with files, we need a place to load them from. If you did read the section in session 1 about running python code, I mentioned that I use both google colab and eclipse pydev for my python development. If you are using eclipse or a similar local IDE then loading or saving is no problem since you have access to your HDD and you can load/save as you like. But if you are using google colab what should you do?

This [link](https://colab.research.google.com/notebooks/io.ipynb "IO colab") shows all the different options you have to deal with files in colab, I tend to use the google drive option.


## 3. Designing a Neural network model for chart recognition

   We prepared our training data, now we can go ahead and create the model we will try to train, this will be no different than what we did before in session one with one exception that the model will be a bit bigger (a lot bigger actually)
   
  ```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.nn import relu,softmax
from tensorflow.keras.optimizers import SGD
model = Sequential()
model.add(Flatten(input_shape=(100,100,3))
model.add(Dense(1024,activation=relu))
model.add(Dense(512,activation=relu))
model.add(Dense(2,activation=softmax))

model.compile(optimizer=SGD(),
               loss='categorical_crossentropy',
               metrics=['accuracy'])
   ```
   
   You will notice that I am using more nodes in each layer than before, and I set my input to be an array of (100,100,3) as we explained before, but neural net Dense layer expects a vector, that is why we use the layer type Flatten to convert our input to a vector.
	You will also notice that my output layer has 2 nodes since i have 2 classes of output to identify, Bar chart and Pie chart, and the loss function I am using here is the categorical crossentropy since my output is more than one class.
	To get an idea of how big is this model, use the summary function to display it, like this:
   
   ```python      
 model.summary()
   ```
   The output you will get when you run this code is 
   
<img src="images/nnsummary.png" height="300" width="500">

As you can see this model has 31M parameters to train. To train the model use this code


```python
model.fit(train_generator,epochs=500,steps_per_epoch=20)
```
   
## 4. Testing model performance

   We loaded our data, designed our model and trained it. Now we want to see how does it perform.

To do this, we need to understand a few new terms

**True Positives (TP):** 
   
   These are the correctly predicted positive values which mean that the value of the actual class is yes and the value of the predicted class is also yes. E.g. if the actual class value is Bar charts, and predicted class is Bar chart.

**True Negatives (TN):** 
   
   These are the correctly predicted negative values which mean that the value of the actual class is no and the value of the predicted class is also no. E.g. if the actual class is not a Bar chart and predicted class tells you the same thing.

**False Positives (FP):** 
   
   When the actual class is no and predicted class is yes. E.g. if the actual class is a Pie chart but predicted class tells you it is a Bar.

**False Negatives (FN):** 
   
   When the actual class is yes but predicted class in no. E.g. if the actual class is a Bar chart but the predicted class is Pie.

**Precision:** 
 
   Precision is the ratio of correctly predicted instances of a class to the total predicted instances of this class. In other words, out of all predicted bar charts how many were really bar charts. High precision indicates a low false positive rate.
   Precision = TP/TP+FP

**Recall:** 

   Recall is the ratio of correctly predicted instances of a class to the all instances of the actual class. In other words out of all bar charts how many were correctly identified. 
   
   Recall = TP/TP+FN


**F1 Score:**  
   
   F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it's better to look at both Precision and Recall.
   
   F1 Score = 2*(Recall * Precision) / (Recall + Precision)
   
This might sound a bit much, but the more you use these the more it will be clear to you, for now lets just focus on the precision

Calculating these numbers, is a task that we  will do many times to verify our models, so it will be a good idea to have a utility method that does that and we can call it any time we need to:
 
```python
   from sklearn.metrics import confusion_matrix, classification_report   
   def test(generator, model):
        predictions = model.predict_generator(generator)
        row_index = predictions.argmax(axis=1)
        print('Classification Report')
        target_names = generator.class_indices.keys()
        print(classification_report(generator.classes, row_index, target_names=target_names))
```
   In the previous code we create a method that receive a data generator (like the one we created before) and a model then it display the numbers we discussed for each class in the model. 

The First line is just using the model to predict all the values for our testing data by calling ``predict_generator`` on our model and passing it the generator.

Remember that our output layer has 2 nodes (one for each class), this means the prediction we will get back will be an array of n rows where n equals the number of images we are passing to the model and each row will be 2 columns with one float number for each class (the number is the probability of the input image being the class), for example for 2 images it might look like this ``[[0.6 0.1] [0.2 0.8]]``, we want to convert this to a vector of n where n equals the number of test and the value in each row reflect the index of the predicted class.

Calling the method argmax on the predictions array will do this for us, for example, if we call argmax on ``[[0.6 0.1] [0.2 0.8]]`` we will get back ``[0 1]``

   The last step is to calculate the precision, recall, and F1 Score. We do not need to calculate it ourselves, we will use the classification_report function from sklearn module, it will do all the work for us. This function requires the class, the correct classification, the predicted classification and the names of the classes. We explained how to get the prediction vector, the correct classification can be obtained from the generator using generator.classes and the labels of the classes using generator.class_indices.keys().

let's call this function now using our training data and see what do we get

```python
   test(generator=train_generator, model=model)
```
you will get this output

<img src="images/train.png" height="150" width="350">

As you can see, our training precision is 100%, but as we said it is more important to calculate these numbers on data set the model did not see before

Let's create a generator using our testing data

```python
test_data_generator = ImageDataGenerator(preprocessing_function=preprocess_input)
test_generator = test_data_generator.flow_from_directory("../smallData/test",
                                                         target_size=(100,100),
                                                         shuffle=False)
```
It is similar code to what we did with the training set, but we pass it the folder for the testing set instead of the training folder, and we set shuffle to False since we do not want to shuffle or testing set.

now we can call our test function agian but this time with the testing data set and see what do we get

```python
   test(generator=test_generator, model=model)
```

you will get something similar to this output

<img src="images/test.png" height="150" width="350">

You will notice that precision now is 78% instead of 100%, but this is a lot more real measure of our model performance since we are running it on data the model did not see before.

An interesting observation here is our training precision is 100%, while our testing precision is 78%. Although 78% is good, when you see a big difference between model training performance and testing performance this is an indication of a problem called Overfitting.

**Overfitting:**
If our model does much better on the training set than on the test set, then we’re likely overfitting. There are few actions you can take to deal with overfitting:
1- Add more samples to your training set
2- Stop your training earlier
3- Remove useless features, for example, does color matter in your classification, maybe you should switch to grayscale instead of RGB

<p align="center"> 
<img src="images/overfitting.png" height="400" width="600">
</p>   

## 5. Confusion matrix


   We checked our model performance, and found that our precision is on average 78%, one important point to keep in mind is when you have a model that does multi-class classifications, looking at the overall performance can be misleading. It is important to check the performance per class as well, this will allow you to see each class performance. For example in our case you will notice that the performance of bar chart is 79% while pie chart is 78% this is fine, but imagine is you find one of them like 20% this means you need to pay this class more attention. For example, what is the quality of the samples of this class, do we have enough samples for it and so on.

Another, important aspect to keep in mind when dealing with multiple classes classification, if we find that our model performs bad for one class, it is important to check what does it confuse this class with. In other words imagine we were detecting Bar, Pie, and Line charts then we find that our model performs bad with line chart classification, it will be very helpful to know what does it confuse line charts with. For example, does it tend to misclassify line charts as bar charts, does it tend to think that pie charts are line charts and so on? This helps us to decide what actions do we need to take to deal with the issue.

This is where the concept of a confusion metrix become very handy. The easiest way to understand confusion matrix is to visualize it:

<p align="center"> 
<img src="images/confusionmatrix.png" height="300" width="500">
</p>   

A quick glance at the confusion matrix will allow you to compare the predicted classes to the actual, what classes get misclassified and what was the misclassification. We want the Diagonal to have the highest number since this reflects that predication matched the true class.

The module we used before to print our the precision and F1 score, comes with a handy function for showing the confusion matrix. Here is how we can show our confusion matrix

we will modify the test function we did before to look like this  

```python
	from sklearn.metrics import confusion_matrix, classification_report
	def test(generator, model):
		predictions = model.predict_generator(generator)
        	row_index = predictions.argmax(axis=1)
        	target_names = generator.class_indices.keys()
        	print(target_names)
        	print(confusion_matrix(generator.classes, row_index))
        	print('Classification Report')
        	print(classification_report(generator.classes, row_index, target_names=target_names))
```

The change was very simple, we added 2 print statements. One to print the names of all class

```python
target_names = generator.class_indices.keys()
print(target_names)
```
and one that calls the utility to print the confusion matrix

```python
print(confusion_matrix(generator.classes, row_index))
```

this should produce:

<img src="images/output.png" height="200" width="300">
 

## 6. Saving the Model

We saw how to create a model and train it, but we never saved our model. Meaning, every time we try to use the model for prediction we have to retrain it. This was OK when we dealt with models like the ones in session 1, where the model can be trained in a minute or so, but real models take a long time to train it can take days. We really need to save our model so we can continue to train where we stopped or we can just load it and use it for prediction without need to retrain. Fortunately, this is easily done with Tensorflow.

* Saving the entire model
 
```python
model.save('my_model.h5')
```
* Loading the model

```python
model = keras.models.load_model('my_model.h5')
```

To see the full code working, you can either get it form the github folder and run it on your local machine or you can use [this Notebook](https://github.com/mohmiim/MLIntroduction/blob/master/session-2/Session_2_first.ipynb) to see the note book, then click on the Open in Colab button to run it and change the code as you like 


## 7. Assignment

We did train a model to recognize 2 chart types, we have been using the samples in the folder called smallData for our training but I did provide another folder called [data](https://github.com/mohmiim/MLIntroduction/tree/master/session-2/data) it has 5 classes instead of 2.
Try to go back and modify the code we did so far to create a model that recognize these 5 charts, what are your observations, for example, did the accuracy improve or get worse, from the confusion matrix what charts are confused with each other. Can you play with the model setup to improve the accuracy?

[Solution](https://github.com/mohmiim/MLIntroduction/blob/master/session-2/NNRecognizer/NNChartRecognizer.py)    

## 8. Observations and Conclusions

We did go a long way in this session, instead of just approximating a simple Linear function, to building a model that recognizes chart type form an image. But there are a few things we need to think about. The quality of our model is not great, considering that we are trying to recognize only 2 types of charts. We know we are suffering from Overfitting issue that we need to deal with. We could try to increase our training set size, but i am  not doing that for now and starting with small training set in the first place on purpose, Because having more labeled samples is not an easy task, and we will learn in session 4 of easy tricks we can do to deal with this issue. Other Options is to add more nodes to our layers or increase the number of layers, but our model already contains 31M parameters to trains (when you save it if you check the file size it will 100MB+), maybe this is an indication that we should take a different approach. In session 3 we will learn a new layer type called Convolution that will help. This is important because Machine learning is an iterative process, and we need to feel comfortable to understand our model and its limitations, so we can get back re-adjust and try again.

All this being said if you have been following so far you should be proud. You did learn quite a few concepts and built machine learning models, analyzed their results and was able to reason about them. Great Job. �   





      
