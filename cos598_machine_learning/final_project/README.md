## Authors
- Christopher Ewanik <christopher.ewanik@maine.edu>
- Dorothy Harris <dorothy.harris@maine.edu>
- Saher Fatima <saher.fatima@maine.edu>

## tl;dr
1. Comparing XGBoost performance with Optuna for perfect hyperparams
2. Classification: XGBoost vs. Naive Bayes vs SVM (Adult Data Set UC Irvine)
3. Regression: XGBoost vs KRR vs ElasticNet (Auto MPG Data Set UC Irvine)
4. Optuna + ElasticNet is CLEAN
5. XGBoost performed similar to others with small datasets
6. Optimizing SVM can be slow

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

## Results and Discussion
Figures 1 and 2, show the test results for all of the classification models
before and after hyperparameter tuning using Optuna. The greatest accuracy
was achieved with the default XGBoost model with an accuracy of around 87%.
The tuned XGBoost model achieved a 94% training accuracy but ultimately
overfit as there was a fairly significant drop off in test accuracy. The Naive
Bayes model struggled greatly with this dataset. The general theory as to why,
was that the mix of one hot encoded features and continuous features was not
a good match for Gaussian Naive Bayes, which is generally better suited for
only continuous data. There could also be some violations of the independence
assumption in this dataset as each row roughly represents a demographic of the
population. The SVM performed closely to the XGBoost model with a testing
accuracy of 85%. It was interesting to note that tuning the hyperparameters
had no significant influence on this model’s performance. The drawback of
the SVM was the massive training time. Tuning the hyperparameters of the
SVM took over an hour. Meanwhile, the XGBoost model took just over three
minutes. This difference appears to be evidence of the sparsity-aware algorithm
and the hardware optimizations that XGBoost implements. Naive Bayes fits
exceptionally fast, but considering the poor performance, it is not likely a viable
option.
6

![Figure 1](https://github.com/chrisewanik/school_projects/assets/113730877/97449483-15cc-4b0b-9308-57e59b4f3b1a)
![Figure 2](https://github.com/chrisewanik/school_projects/assets/113730877/226f978a-d099-48a5-adad-b2549a830438)

In Figures 3 and 4, the tuned KRR performed best among all models with a low
test MSE of 0.13, although the tuned Elastic Net and XGBoost model did not
lag far behind. It was interesting to see the strength of the default XGBoost
regression although it ultimately lead to overfitting. The Elastic Net regression
greatly benefited from optimizing the L1 and L2 regularization penalties. This
was the most drastic improvement seen with the use of Optuna. All of the
models were optimized and trained in a timely manner, which was expected
given the small number of rows in the dataset.

![Figure 3](https://github.com/chrisewanik/school_projects/assets/113730877/d4c0c5ee-f319-40df-8fb8-a3f52eab2f9b)
![Figure 4](https://github.com/chrisewanik/school_projects/assets/113730877/62ed6d33-090c-47db-8061-9517ff4bd1a5)

## Conclusion

When using Optuna and a Tree-Structured Parzen Estimator algorithm
to tune the hyperparameters, there was no significant evidence that XGBoost
provided meaningful advantages over traditional algorithms on small to medium
sparse datasets. Support Vector Machine took a dramatic amount of time to tune
its hyperparameters and Elastic Net regression benefited greatly from optimizing
the L1 and L2 tradeoff. Optuna and TPE proved effective in improving model
performance, boosting all but one model in testing accuracy. Optuna also proved
highly efficient scanning through huge ranges of hyperparameters that would
have taken dramatically longer with traditional grid searches.



