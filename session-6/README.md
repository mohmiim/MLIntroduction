# Introduction to timeseries forecasting with deeplearning 


## What will we cover

[1. What is Forecasting](#1-what-is-forecasting)

[2. Goal of the session](#2-goal-of-the-session)

[3. Time series attributes](#3-time-series-attributes)

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
