
# Machine learning project: How well do you barbell lifts?

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways:
        
* Class A: exactly according to the specification

* Class B: throwing the elbows to the front

* Class C: lifting the dumbbell only halfway

* Class D: lowering the dumbbell only halfway

* Class E: throwing the hips to the front

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.


### Getting and cleaning data

```r
setwd("C:/Users/m.hernandez.moreno/Desktop/Curso Data Science/08. Practical Machine Learning/Project")

# Loading the training and testng data set
trainData <- read.csv("pml-training.csv")
testData <- read.csv("pml-testing.csv")

dim(trainData)
```

```
## [1] 19622   160
```

```r
dim(testData)
```

```
## [1]  20 160
```

Of those 160 variables, select ones with high (over 95%) missing data and exclude them from the analysis:

```r
trainData[trainData==""] <- NA
NArate <- apply(trainData, 2, function(x) sum(is.na(x)))/nrow(trainData)
trainData <- trainData[!(NArate>0.95)]

testData[testData==""] <- NA
NArate <- apply(testData, 2, function(x) sum(is.na(x)))/nrow(testData)
testData <- testData[!(NArate>0.95)]

dim(trainData)
```

```
## [1] 19622    60
```

Remove variables have near zero variance and remove them from the data:

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.5.1
```

```
## Loading required package: ggplot2
```

```r
nzv <- nearZeroVar(trainData, saveMetrics = TRUE)
trainData <- trainData[,nzv$nzv == "FALSE"]
trainData$classe <- as.factor(trainData$classe)

nzv <- nearZeroVar(testData, saveMetrics = TRUE)
testData <- testData[, nzv$nzv == "FALSE"]

dim(trainData)
```

```
## [1] 19622    59
```

```r
colnames(trainData)
```

```
##  [1] "X"                    "user_name"            "raw_timestamp_part_1"
##  [4] "raw_timestamp_part_2" "cvtd_timestamp"       "num_window"          
##  [7] "roll_belt"            "pitch_belt"           "yaw_belt"            
## [10] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
## [13] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [16] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [19] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [22] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [25] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [28] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [31] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [34] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [37] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [40] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [43] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [46] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [49] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [52] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [55] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [58] "magnet_forearm_z"     "classe"
```

Delete from variable 1 to variable 6 because they can't be predictors:

```r
trainData <- trainData[,-c(1:6)]
testData <- testData[, -c(1:6)]
dim(trainData)
```

```
## [1] 19622    53
```

Our new cleaned data set has **53 variables.**


### Cross validation: 60% training 40% testing

```r
set.seed(1122)
inTrain <- createDataPartition(trainData$classe,p = 0.6,list=FALSE)
training <- trainData[ inTrain,]
testing <- trainData[-inTrain,]
```


### Training Random Forest

```r
library(randomForest)
```

```
## Warning: package 'randomForest' was built under R version 3.5.1
```

```
## randomForest 4.6-14
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
set.seed(1122)
rf <- randomForest(classe ~ .,data=training)
print(rf)
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = training) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.59%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 3346    1    0    1    0 0.0005973716
## B   12 2261    6    0    0 0.0078982010
## C    0   14 2037    3    0 0.0082765336
## D    0    0   22 1905    3 0.0129533679
## E    0    0    2    6 2157 0.0036951501
```

Let's check the accuracy and the sample error with testing data:


```r
predictTestingRF <- predict(rf, testing)
confusionMatrix(testing$classe, predictTestingRF)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2223    6    0    0    3
##          B    4 1507    7    0    0
##          C    0   22 1344    2    0
##          D    0    0   21 1263    2
##          E    0    0    4    1 1437
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9908          
##                  95% CI : (0.9885, 0.9928)
##     No Information Rate : 0.2838          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9884          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9818   0.9767   0.9976   0.9965
## Specificity            0.9984   0.9983   0.9963   0.9965   0.9992
## Pos Pred Value         0.9960   0.9928   0.9825   0.9821   0.9965
## Neg Pred Value         0.9993   0.9956   0.9951   0.9995   0.9992
## Prevalence             0.2838   0.1956   0.1754   0.1614   0.1838
## Detection Rate         0.2833   0.1921   0.1713   0.1610   0.1832
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9983   0.9900   0.9865   0.9971   0.9979
```


```r
accuracyRF <- confusionMatrix(testing$classe, predictTestingRF)$overall[1]
cat("Accuracy: ", accuracyRF)
```

```
## Accuracy:  0.9908233
```

```r
seRF <- 1 - confusionMatrix(testing$classe, predictTestingRF)$overall[1]
cat("Out of sample error: ", seRF)
```

```
## Out of sample error:  0.009176651
```


### Linear Discriminant Analysis


```r
library(caret)

set.seed(1122)
lda <- train(classe ~ ., method = "lda",data=training)
print(lda)
```

```
## Linear Discriminant Analysis 
## 
## 11776 samples
##    52 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Bootstrapped (25 reps) 
## Summary of sample sizes: 11776, 11776, 11776, 11776, 11776, 11776, ... 
## Resampling results:
## 
##   Accuracy   Kappa    
##   0.7013024  0.6218096
```

Let's check the accuracy with testing data:

```r
predictTestingLDA <- predict(lda, testing)
confusionMatrix(testing$classe, predictTestingLDA)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1813   51  178  183    7
##          B  236  944  209   64   65
##          C  156  136  882  157   37
##          D   75   48  153  950   60
##          E   48  277  131  153  833
## 
## Overall Statistics
##                                           
##                Accuracy : 0.6911          
##                  95% CI : (0.6807, 0.7013)
##     No Information Rate : 0.2967          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.609           
##  Mcnemar's Test P-Value : < 2.2e-16       
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.7788   0.6484   0.5679   0.6304   0.8313
## Specificity            0.9241   0.9102   0.9228   0.9470   0.9110
## Pos Pred Value         0.8123   0.6219   0.6447   0.7387   0.5777
## Neg Pred Value         0.9083   0.9191   0.8964   0.9151   0.9736
## Prevalence             0.2967   0.1856   0.1979   0.1921   0.1277
## Detection Rate         0.2311   0.1203   0.1124   0.1211   0.1062
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.8514   0.7793   0.7454   0.7887   0.8712
```


```r
accuracyLDA <- confusionMatrix(testing$classe, predictTestingLDA)$overall[1]
cat("Accuracy: ", accuracyLDA)
```

```
## Accuracy:  0.6910528
```


```r
seLDA <- 1- confusionMatrix(testing$classe, predictTestingLDA)$overall[1]
cat("Out of sample error: ", seLDA)
```

```
## Out of sample error:  0.3089472
```


**Random forest method is much more accurate that linear discriminant analysis, and the out of sample error of this method is very small, 0.91%**


```r
library(rpart)
```

```
## Warning: package 'rpart' was built under R version 3.5.1
```

```r
library(rpart.plot)
```

```
## Warning: package 'rpart.plot' was built under R version 3.5.1
```

```r
rfModelTree <- rpart(classe ~., data = training, method="class")
prp(rfModelTree,main="Classification Tree")
```

![plot of chunk unnamed-chunk-14](figure/unnamed-chunk-14-1.png)

### Run the Prediction Model on the Test Data
Finally, use the random forest prediction model to predict 20 different test cases:

```r
predict(rf, testData)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

### Conclusions

In this project has been appliyed cross-validation in traning data (19622 observations), 70% of the total observations to build a prediction model, and the rest of 30% of the observations to model validation.
Comparing two prediction models, random forest method and linear discriminat analysis, it's concluded that random forest method is much more accurance to predict test cases (accuracy of 99,09% and out of error sample to 0,91%)
The model statistics of this method showed that the sensitivity was in between 97%-99% and the specificity was over 99% for all classes.
In conclusion, the model is well developed to predict the exercise classes during barbell lifts.
