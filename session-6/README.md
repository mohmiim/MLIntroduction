# Introduction to time series forecasting with deeplearning 


## What will we cover

[1. What is Forecasting](#1-what-is-forecasting)

[2. Goal of the session](#2-goal-of-the-session)

[3. Time series attributes](#3-time-series-attributes)

[4. Sequence Models](#4-sequence-models)

[5. Picking our dataset](#5-picking-our-dataset)

[6. Prepare Data for training Tensorflow model](#6-prepare-data-for-training-tensorflow-model)

[7. Design our Tensorflow model](#7-design-our-tensorflow-model)

[8. Putting it all together](#8-putting-it-all-together)

[9. Conclusion](#9-conclusion)



## 1. What is Forecasting

Forecasting, is the act of using historical data as an input to learn and make predictions about the direction of future trends. 
Formal definition if Y = {y<sub>1</sub>,y<sub>2</sub>, .... y<sub>n</sub>}then forecasting is finding the values of y<sub>n+h</sub> where h is the horizon of the forecast
It is important to keep in mind that it is important to know what type of forecasting you are trying to do  
**Univariate time Series:** where only one variable is changing over time, for example if we have a sensor recording temperature over time   
**Multivariate time series:** more than one variable is changing over time, for example accelerometer where X,Y and Z are recorded over time 

## 2. Goal of the session

Our goal is to show how can we build a deeplearning model to do forecasting of Multivariate time series?. I will not get in the discussion about deeplearning VS classical statistics for time series forecasting. Quick search will show that it is being debated even for univariate time series with papers supporting both sides, so i will leave that to you as a self reading after you finish the tutorial 

## 3. Time series Attributes

Before we jump into the Machine Learning aspect of this tutorial it is important to make sure we understand few basic attributes of time series  

**trend:** a systematic linear or nonlinear component that changes over time and does not repeat  

<p align="center"> 
<img src="images/trend.png" height="300">
</p>


**Seasonality:** a systematic linear or nonlinear component that changes over time and repeat  

<p align="center"> 
<img src="images/seasonal.png" height="300">
</p>

In real life data you will find both trend and seasonality together

<p align="center"> 
<img src="images/both.png" height="300">
</p>

**Noise:** non-systematic component that is neither trend nor seasonal

<p align="center"> 
<img src="images/noise.png" height="300">
</p>

Typical time series will contain all three

<p align="center"> 
<img src="images/all.png" height="300">
</p>

## 4. Sequence Models

Time to start talk Machine Learning, In the previous sessions we did learn many types of Machine learning layers and models, like Neural Networks and Convolution Neural Networks (CNN). Time series will require us to learn a new type or Models which is called sequence models. The name almost till it all, in these models order does matter, because the data it is trying to learn are order sensitive. This is important in many applications for example in Natural Language Processing where the order or words affects how we understand the sentence, also in time series analysis without the correct order our analysis will be invalid.

The most basic type of sequence models is Recurrent Neural Networks, where the output of the previous step impact the out of the next step

The next diagram, show how the output of the previous step is fed into the next step

<p align="center"> 
<img src="images/RNN.png" height="300">
</p>

This is the base idea of sequence models, but we will not really use RNNs because there are cases where RNN will suffer. These cases are the ones where the output depends on context very early in the sequence. RNN can be adjust to handle cases like these but there is a nother type of network that can handle these cases much easier this type is Long Short Term Memory in short LSTM

**LSTM:**

LSTM are a variation of RNN, that deal specifically with the long-term dependency problem because it remember information for long time.

The details of how LSTMS works is beyond the scope of this tutorial and i suggest goign through the [amazing deeplearing specialization by Andrew NG](https://www.deeplearning.ai/deep-learning-specialization/) on coursera to get in the details (course 5)  

## 5. Picking our dataset  

We mentioned in previous section that we can have univariate or multivariate forecasting and the difference between both. So we will pick 2 data sets to tackle one is univariate and one is multivariate

**[Sunspots:](https://github.com/mohmiim/MLIntroduction/tree/master/session-6/data/Sunspots.csv)**  
Univarate data set that contains the monthly sunspot data since 1749

<p align="center"> 
<img src="images/sunspots.png" height="300">
</p>

**[Individual household electric power consumption:](https://archive.ics.uci.edu/ml/datasets/individual+household+electric+power+consumption)** 
Multivariate dataset made available by [UCI Machine Learning repository](https://archive.ics.uci.edu/ml/datasets.php), besside being a Multivariate dataset it had some missing values which is good because this is what you will ctually find in real life datasets.

<p align="center"> 
<img src="images/energy.png" height="300">
</p>

## 6. Prepare Data for training Tensorflow model

What does it mean to prepare data for training ?. Lets consider the simplest form of a time series 

~~~~{.python}
1  
2  
3  
4  
5  
6  
7
~~~~

This is just a sequence of numbers, but as we have seen in the previous sessions to train a model we need input and output so how can we convert a list like that to a training set?


~~~~{.python}
[1 2]
[2 3]
[3 4]
[4 5]
[5 6]
[6 7]  
~~~~

when we do that we did convert our input to a set of sequences, we can doa similar thing to the output where the output will be the next item in the original sequence

~~~~{.python}
[1 2] [3]
[2 3] [4]
[3 4] [5]
[4 5] [6]
[5 6] [7]
~~~~

Tensor flow actually provide some nice utilities to make it easy to do that 
lets look at some code

- first we let's create our simple list of numbers

~~~~{.python}
import tensorflow as tf
dataset = tf.data.Dataset.range(10)
for val in dataset: print(val.numpy())
~~~~
 
The out put of this code should be 

~~~~{.python}
0
1
2
3
4
5
6
7
8
9
~~~~

- next we want to createthe sliding window 

~~~~{.python}
dataset = dataset.window(2, shift=1, drop_remainder=True)
dataset = dataset.flat_map(lambda window: window.batch(2))
for window in dataset:
	print(window.numpy())
~~~~

We used the window function on the dataset, we pass it how long is the window and how many numbers it should shift between each window, then the drop_remainder flag determines what will happen if we have partial windows at the end, should they be kept or dropped.
After that we call the numpy method to convert it to numpy arrays
 
the output of this will be 

~~~~{.python}
[0 1 2]
[1 2 3]
[2 3 4]
[3 4 5]
[4 5 6]
[5 6 7]
[6 7 8]
[7 8 9]
~~~~

the last step is to convert it make it X , y for training and labels. Which is done using simple array API's like that :

~~~~{.python}
dataset = dataset.map(lambda window: (window[:-1], window[-1:]))
for X,y in dataset:
	print(X.numpy(), y.numpy())
~~~~

The output will look like this 

~~~~{.python}
[0 1] [2]
[1 2] [3]
[2 3] [4]
[3 4] [5]
[4 5] [6]
[5 6] [7]
[6 7] [8]
[7 8] [9]
~~~~
 
## 7. Design our Tensorflow model  
## 8. Putting it all together  
## 9. Conclusion
