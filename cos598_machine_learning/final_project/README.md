## tl;dr
1. Optuna + ElasticNet is highly effective
2. XGBoost performed similar to others with small datasets
3. Optimizing SVM can be slow

## Abstract

XGBoost is considered one of the leading machine learning algorithms
for regression and classification and is used by data scientists worldwide.
Due to its high accreditation, this study tests XGBoost’s performance
against its common academia counterparts by both performance and training
time. This study was conducted by comparing XGBoost’s classification
model to Support Vector Classification (SVM) and Naive Bayes, and
XGBoost’s regression model to Kernel Ridge Regression (KRR) and Elastic
Net. The hyperparameters are tuned with a Tree-Structured Parzen
Estimator (TPE) using a library called Optuna. For classification, the
default XGBoost had the best testing accuracy at 0.87. For Regression,
the Optuna tuned KRR had the best testing MSE at 0.13. The TPE was
highly effective for tuning hyperparameters, particularly with the Elastic
Net Regression. The SVM took over an hour to optimize hyperparameters
and, despite more parameters, XGBoost optimized relatively quickly.

## Introduction

This project aimed to compare the speed and performance of XGBoost with traditional academic machine learning methods. 
The paper claimed that XGBoost's algorithm runs over ten times faster than existing popular solutions, but it had limitations in terms of dataset size and comparison scope.
The study's motivation was to investigate whether XGBoost provided competitive advantages over traditional algorithms, particularly on small to medium datasets. 
The focus was on XGBoost's performance and speedup capabilities for classification and regression problems, comparing it to classic machine learning methods such as SVM, Naive Bayes, 
Elastic Net, and KRR. The hyperparameters of all models were tuned using Optuna. Two datasets from UC Irvine were utilized, the Adult dataset for classification models and the 
Auto MPG dataset for regression models, each chosen based on their characteristics and suitability for comparison.


## Data
XGBoost claims to have superior performance on larger sparse datasets created
from feature engineering. Therefore, this paper preprocesses the data in a way
that could potentially show performance differences between different algorithms
and XGBoost. Overall, all rows with NAs are first removed, as well as any
redundant columns. Next, a standard scalar is applied to the continuous variables.
In the auto data set, the car name is split into "make" and "model". The most
significant change is using one-hot encoding on all categorical columns. This
results in the addition of more dimensions that contain lots of sparse data.
The adult dataset goes from 32,561 rows and 15 columns to 30,162 rows and
103 columns, while the auto dataset changes from 390 rows with 10 columns
5
to 390 rows with 26 columns. These choices were made with the hope of
testing XGBoost’s ability with smaller datasets and challenging the traditional
algorithm’s ability to deal with sparsity. Data is split into training and testing
sets using 30% of the data for testing.

## Methods

The study aimed to evaluate the training performance and time of various machine learning models in Python. 
The models used included Support Vector Classification, ElasticNet Regression, Naive Bayes, Kernel Ridge Regression, XGBoost Regression, and XGBoost Classification. 
The study employed the Optuna optimization library and specifically used the Tree-Structured Parzen Estimator Algorithm, to optimize the models' hyperparameters.

For classification tasks, the study used a SVM and Naive Bayes to compare with XGBoost.
For regression, we chose to use ElasticNet Regression and Kernel Ridge Regression to test XGBoost Regression. 
The study measured both the training speed and model performance of the different algorithms. To quantify the classification performance of XGBoost, 
it was tested against Support Vector Classification and Naive Bayes. For regression performance, XGBoost Regression was compared to ElasticNet Regression and Kernel Ridge Regression.

