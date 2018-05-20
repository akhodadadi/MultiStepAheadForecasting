# MultiStepAheadForecasting
multi-step ahead forecasting of spatio-temporal data

# Introduction
The goal of this project is to investigate two aspects of multi-step-ahead forecasting of spatio-temporal data:

1. *Dynamic vs static models:* we will compare the performance of several static and dynamic models. The dynamic models all have a recurrent neural network as part of their architecture. In these models, the value of the time-series in the previous time-steps is used to derive the "state" of the recurrent netwok. The output of the recurrent network, then, is augmented to other features in the data to form the full set of features. In contrast, in the static models there is no recurrent archtucture and the value of the time-series at previous time-steps are augmented directly to the other features. 

2. *methods for splitting data:* we will examine the effect of different methods for splitting the data into train and validation on the performance of the trained model on the test data. Forming the validation set for time-seris data is more challenging than other cases. Specifically, many machine learning tasks can be considered as *interpolation*, where the range of the features in the train and test sets are similar. On the other hand, time-series forecasting (specially the multi-step ahead forecasting) is an *extrapolation* task. The question that we are intrested in is wheather this should be considered in forming the validation set. We will examine different methods for forming the valifation set.

We will examine these points on the [Recruit dataset](https://www.kaggle.com/c/recruit-restaurant-visitor-forecasting). This is a relational dataset which includes the daily number of visitors to a number of restaurants in Japan from 2016-01-01 to 2017-04-22. The goal is to forecast the number of visitors for these restaurants for the interval 2017-04-23 to 2017-05-31 (38 days). The quality of forecast is measured by root mean squared logarithmic error (RMSLE) defined as follows:

$$
RMSLE=\sqrt{\frac{1}{n} \sum_i (log(y_i+1)- log(\hat{y}_i+1))^2}
$$

where $y_i$ and $\hat{y}_i$ are the $i^{th}$ observation and estimated value, respectively.

In addition to the number of visitors, the dataset contains information about the longitude and latitude of the stores, the genre of the food they serve, the name of the neighberhood and the status of the date (holiday or not). These data are in three tables: visit_data (which contains number of visitors, store id and the date), store_info (which contains the information regariding the stores) and date_info (which contains information about the dates). There are a total of 252108 observation in the visti_data table which belongs to 829 stores.
