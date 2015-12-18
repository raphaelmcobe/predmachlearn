---
title: "Practical Machine Learning Course Project"
author: "Raphael Cóbe"
date: "18 de dezembro de 2015"
output: 
  html_document: 
    keep_md: yes
    toc: yes
---

# Executive Summary
The goal of this project is to predict the manner in which they did the exercise. The dataset used was the one from “Weight Lifting Exercise Dataset” from PUC-Rio. The goal was to, accurately predict the "classe" variable in the training set.

# Prerequisites
Bellow one will find the import section of the libraries used to develop the project:

```r
library(doMC)
registerDoMC(cores = 4)
library(caret)
library(gbm)
library(rpart)
library(randomForest)
set.seed(999)
```


# Getting and Cleaning Data

First we need to load both the training and the test dataset. Then we divide the training dataset into a training partition (75% of the data - in sample) and a test partition (25% of the date - out-of-sample) in order to test our model with a larger number of case to predict.


```r
dataset.training <- read.csv("pml-training.csv")
dataset.test <- read.csv("pml-testing.csv")
training_partition <- createDataPartition(y=dataset.training$classe, p=0.75, list=FALSE)
training <- dataset.training[training_partition,]
test <- dataset.training[-training_partition,]
```

After that we noticed that the dataset contained a large number of variables with several NA values on them. Then we went through a process of feature selection, discarding the variables that wouldn't help building the model as explained on the next session.

# Selecting Features

First we had to remove the variables remove X, user_name, *_timestamp, *_window since they are qualitative variables and wouldn't help much in building a model. These variables are stored in columns 1 to 7.


```r
training <- training[,8:160]
```

Then we've had to remove the ones with **Near Zero Covariance** and also remove the columns that contains NAs values:


```r
nzv <- nearZeroVar(training, saveMetrics=TRUE)
training <- training[,!nzv[,4]]
training <- training[, colSums(is.na(training)) == 0]
```

After that we produced a clean dataset containing 53 variables.

Our approach consisted basically to test a few methods for building models and using them to predict the outcome of our test partition.
The first one to be used was the Decision Trees.

# Predicting with Decision Trees

We used the caret package to train our decision tree model:

```r
model <- train(classe ~ ., data=training, method="rpart")
```

We verify our model with our own training data partition.

```r
pred <- predict(model, training)
confusionMatrix(pred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 3810 1178 1190 1078  399
##          B   60  955   76  413  370
##          C  304  715 1301  921  716
##          D    0    0    0    0    0
##          E   11    0    0    0 1221
## 
## Overall Statistics
##                                          
##                Accuracy : 0.4951         
##                  95% CI : (0.487, 0.5032)
##     No Information Rate : 0.2843         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.3402         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9104  0.33532   0.5068   0.0000  0.45122
## Specificity            0.6350  0.92258   0.7814   1.0000  0.99908
## Pos Pred Value         0.4977  0.50961   0.3288      NaN  0.99107
## Neg Pred Value         0.9469  0.85262   0.8824   0.8361  0.88989
## Prevalence             0.2843  0.19350   0.1744   0.1639  0.18386
## Detection Rate         0.2589  0.06489   0.0884   0.0000  0.08296
## Detection Prevalence   0.5201  0.12733   0.2689   0.0000  0.08371
## Balanced Accuracy      0.7727  0.62895   0.6441   0.5000  0.72515
```

With that we've discovered the the accuracy of our model was only of 49%, wich is a very low value. Then, we decided to use a different model generation strategy. We tryed then the General Boosting Method


# Predicting with Boosting

In order to perform this experiment we've also used the caret package.

```r
model <- train(classe ~ ., data=training, method="gbm")
```

```
## Iter   TrainDeviance   ValidDeviance   StepSize   Improve
##      1        1.6094            -nan     0.1000    0.2380
##      2        1.4598            -nan     0.1000    0.1614
##      3        1.3571            -nan     0.1000    0.1193
##      4        1.2806            -nan     0.1000    0.1068
##      5        1.2117            -nan     0.1000    0.0887
##      6        1.1554            -nan     0.1000    0.0685
##      7        1.1112            -nan     0.1000    0.0635
##      8        1.0708            -nan     0.1000    0.0719
##      9        1.0264            -nan     0.1000    0.0644
##     10        0.9853            -nan     0.1000    0.0488
##     20        0.7571            -nan     0.1000    0.0232
##     40        0.5313            -nan     0.1000    0.0098
##     60        0.4059            -nan     0.1000    0.0098
##     80        0.3262            -nan     0.1000    0.0056
##    100        0.2673            -nan     0.1000    0.0035
##    120        0.2243            -nan     0.1000    0.0026
##    140        0.1925            -nan     0.1000    0.0023
##    150        0.1784            -nan     0.1000    0.0010
```

The results, trying to predict the outcome of our training partition (in-sample) were of 97.3% accuracy, which suggests a good model.


```r
pred <- predict(model, training)
confusionMatrix(pred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4143   68    0    1    4
##          B   28 2732   50    5   12
##          C    8   44 2486   53   11
##          D    4    4   28 2342   23
##          E    2    0    3   11 2656
## 
## Overall Statistics
##                                         
##                Accuracy : 0.9756        
##                  95% CI : (0.973, 0.978)
##     No Information Rate : 0.2843        
##     P-Value [Acc > NIR] : < 2.2e-16     
##                                         
##                   Kappa : 0.9691        
##  Mcnemar's Test P-Value : 1.924e-08     
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9900   0.9593   0.9684   0.9710   0.9815
## Specificity            0.9931   0.9920   0.9905   0.9952   0.9987
## Pos Pred Value         0.9827   0.9664   0.9554   0.9754   0.9940
## Neg Pred Value         0.9960   0.9902   0.9933   0.9943   0.9958
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2815   0.1856   0.1689   0.1591   0.1805
## Detection Prevalence   0.2865   0.1921   0.1768   0.1631   0.1815
## Balanced Accuracy      0.9915   0.9756   0.9794   0.9831   0.9901
```

After that we felt confortable to evaluate our model by trying to predict the classe variable at our test partition (Out-of-sample):

```r
pred_test <- predict(model, test)
confusionMatrix(pred_test, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1373   37    0    0    2
##          B   16  890   26    2    7
##          C    4   19  819   26   11
##          D    2    3    9  770    6
##          E    0    0    1    6  875
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9639         
##                  95% CI : (0.9583, 0.969)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9543         
##  Mcnemar's Test P-Value : 1.04e-05       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9842   0.9378   0.9579   0.9577   0.9711
## Specificity            0.9889   0.9871   0.9852   0.9951   0.9983
## Pos Pred Value         0.9724   0.9458   0.9317   0.9747   0.9921
## Neg Pred Value         0.9937   0.9851   0.9911   0.9917   0.9935
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2800   0.1815   0.1670   0.1570   0.1784
## Detection Prevalence   0.2879   0.1919   0.1792   0.1611   0.1799
## Balanced Accuracy      0.9866   0.9625   0.9715   0.9764   0.9847
```

The results suggest an accuracy of roughly 96%.

We were curious to try and use a different model building technique in order to check if we could enhance further our accuracy.

# Predicting with Random Forests

After experimenting with the general boosting we tried to use the Random Forest approach.

As the number of interactions for the default train function is quite high, we imposed a few constraints to our model training approach by feeding the train function with a TrainControl strategy. In that strategy we used a 8-fold cross valisation strategy. The number 8 was aligned with the number of cpus on the machine performing the experiment.


```r
model <- train(classe ~ ., data=training, method="rf", 
               trControl=trainControl(method="cv", number=8))
```

Using this approach we were able to achieve a prediction accuracy (using the train partition - in sample) of 100%, which indicates that this model can predict the classe variable in all cases.

```r
pred <- predict(model, training)
confusionMatrix(pred, training$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 4185    0    0    0    0
##          B    0 2848    0    0    0
##          C    0    0 2567    0    0
##          D    0    0    0 2412    0
##          E    0    0    0    0 2706
## 
## Overall Statistics
##                                      
##                Accuracy : 1          
##                  95% CI : (0.9997, 1)
##     No Information Rate : 0.2843     
##     P-Value [Acc > NIR] : < 2.2e-16  
##                                      
##                   Kappa : 1          
##  Mcnemar's Test P-Value : NA         
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   1.0000   1.0000   1.0000   1.0000
## Specificity            1.0000   1.0000   1.0000   1.0000   1.0000
## Pos Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Neg Pred Value         1.0000   1.0000   1.0000   1.0000   1.0000
## Prevalence             0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Rate         0.2843   0.1935   0.1744   0.1639   0.1839
## Detection Prevalence   0.2843   0.1935   0.1744   0.1639   0.1839
## Balanced Accuracy      1.0000   1.0000   1.0000   1.0000   1.0000
```

We then tried to predict the classe variable of our test data partition (out-of-sample) and the achieved accuracy was of 99%. 

```r
pred_test <- predict(model, test)
confusionMatrix(pred_test, test$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1395    5    0    0    0
##          B    0  942    4    0    0
##          C    0    2  851   14    0
##          D    0    0    0  789    1
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9945         
##                  95% CI : (0.992, 0.9964)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.993          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            1.0000   0.9926   0.9953   0.9813   0.9989
## Specificity            0.9986   0.9990   0.9960   0.9998   0.9998
## Pos Pred Value         0.9964   0.9958   0.9815   0.9987   0.9989
## Neg Pred Value         1.0000   0.9982   0.9990   0.9964   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2845   0.1921   0.1735   0.1609   0.1835
## Detection Prevalence   0.2855   0.1929   0.1768   0.1611   0.1837
## Balanced Accuracy      0.9993   0.9958   0.9957   0.9905   0.9993
```

After that we felt confident in using the test dataset provided.
# Predicting the testing dataset


```r
final <- read.csv("pml-testing.csv")
answers <- as.character(predict(model, final))
answers
```

```
##  [1] "B" "A" "B" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```

# Conclusion

We were able to build a model capable of predicting correctly 100% of our in-sample data and roughly 99% of our out-of-sample data with a 95%-confidence interval of (0.992, 0.9964). That could only be achieved by using the parallel capabilities of the caret package which speeded-up the process of training the models. Also the values of Specificity values were higher than 0.99 for all prediction classes and the Sesitivity measures were also high for all classes (0.98 and higher).
