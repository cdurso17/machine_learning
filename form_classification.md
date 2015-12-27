---
title: "Classifying Exercise Form"
author: "cdurso"
date: "Sunday, December 27, 2015"
output: html_document
---

## Summary

The goal is to develop a machine-learning algorithm that will identify which of 5 forms is being used for a motion by a subject based on multiple sensor measurements at a single time point during the motion. 

A set of training data and a set of test data were provided. The training data consist of a collection of time-series measurements from the sensors monitoring an individual performing an exercise using correct form or using one of 4 specified incorrect forms. The type of form is identified. The test data consist of a collection of measurements from the sensor array at a single time point during an exercise. The type of form is not identified.

A random forest model based on 31 principal components of the raw sensor data has a five-fold cross-validation accuracy of 98%.

## Data Entry and Cleaning

Columns with no valid data must be removed from the test data. The corresponding columns must be removed from the training data.


```r
## Load required packages
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
library(randomForest)
```

```
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
```

```r
dat<-read.csv("pml-training.csv",na.strings=c("#DIV/0!","NA"))
testdat<-read.csv("pml-testing.csv",na.strings=c("#DIV/0!","NA"))

# Drop columns that are predominantly "NA"
ok<-apply(dat,2,function(x){sum(is.na(x))/length(x)})<.8
testok<-apply(testdat,2,function(x){sum(is.na(x))/length(x)})<.8

datok<-dat[,ok]
testdatok<-testdat[,testok] 

dat<-datok
testdat<-testdatok
```

Because the class frequency varies by user_name, user_name should be removed as a variable. Also, data relating to the time series must be removed from the training set, because they aren't available in the test data.



```r
trim<-dat[,8:60]
testtrim<-testdat[,8:59]
```

Principal components will be used for the analysis, so must be built in the training data, and the same transformation applies to the test data. Analysis done elsewhere indicates that 45 principal components fully train the model.


```r
pp<-preProcess(trim[,1:52],method="pca",pcaComp=45,thresh=0.9999)
trim.pr<-predict(pp, trim)
test.pr<-predict(pp, testtrim)
```

## Model Fitting

The random forest model on 45 principal components has high accuracy, over 98%.


```r
set.seed(12345)
trimModel.pr<-randomForest(classe ~ ., data=trim.pr) # Fit the model.
confusionMatrix(trimModel.pr$predict,trim.pr$classe) # Get an in-sample error summary.
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 5575   48    5    4    2
##          B    2 3724   29    0    5
##          C    0   19 3375   94   14
##          D    2    3   12 3112    9
##          E    1    3    1    6 3577
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9868          
##                  95% CI : (0.9851, 0.9884)
##     No Information Rate : 0.2844          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9833          
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9991   0.9808   0.9863   0.9677   0.9917
## Specificity            0.9958   0.9977   0.9922   0.9984   0.9993
## Pos Pred Value         0.9895   0.9904   0.9637   0.9917   0.9969
## Neg Pred Value         0.9996   0.9954   0.9971   0.9937   0.9981
## Prevalence             0.2844   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2841   0.1898   0.1720   0.1586   0.1823
## Detection Prevalence   0.2871   0.1916   0.1785   0.1599   0.1829
## Balanced Accuracy      0.9975   0.9892   0.9892   0.9830   0.9955
```

## Cross-validation

Five-fold cross-validation estimates the out-of-sample accuracy at 98%.


```r
set.seed(12345)

## Set up indices for 5 folds.
folds <- createFolds(y=trim.pr$classe,k=5,list=TRUE,returnTrain=TRUE)
tests<-lapply(folds,function(x){setdiff(1:length(trim.pr$classe),x)})
  	rffold<-lapply(folds,
		function(x){randomForest(classe ~ ., data=trim.pr[x,])})
## Cycle through the folds, totaling the prediction errors.
errorct<-0
for(i in 1:5){
	errorct<-errorct+
	sum(predict(rffold[[i]],newdata=trim.pr[tests[[i]],])!=
	trim.pr[tests[[i]],]$classe)
}
1-errorct/nrow(trim.pr) ## This calculates the accuracy of the cross-validated predictions.
```

```
## [1] 0.983437
```

