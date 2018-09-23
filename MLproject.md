
# Machine learning project: How well do you barbell lifts?

One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways:
        
* Class A: exactly according to the specification

* Class B: throwing the elbows to the front

* Class C: lifting the dumbbell only halfway

* Class D: lowering the dumbbell only halfway

* Class E: throwing the hips to the front

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.


### Getting and cleaning data
```{r}
setwd("C:/Users/m.hernandez.moreno/Desktop/Curso Data Science/08. Practical Machine Learning/Project")

# Loading the training and testng data set
trainData <- read.csv("pml-training.csv")
testData <- read.csv("pml-testing.csv")

dim(trainData)
dim(testData)
```

Of those 160 variables, select ones with high (over 95%) missing data and exclude them from the analysis:
```{r}
trainData[trainData==""] <- NA
NArate <- apply(trainData, 2, function(x) sum(is.na(x)))/nrow(trainData)
trainData <- trainData[!(NArate>0.95)]

testData[testData==""] <- NA
NArate <- apply(testData, 2, function(x) sum(is.na(x)))/nrow(testData)
testData <- testData[!(NArate>0.95)]

dim(trainData)
```

Remove variables have near zero variance and remove them from the data:
```{r}
library(caret)
nzv <- nearZeroVar(trainData, saveMetrics = TRUE)
trainData <- trainData[,nzv$nzv == "FALSE"]
trainData$classe <- as.factor(trainData$classe)

nzv <- nearZeroVar(testData, saveMetrics = TRUE)
testData <- testData[, nzv$nzv == "FALSE"]

dim(trainData)
colnames(trainData)
```

Delete from variable 1 to variable 6 because they can't be predictors:
```{r}
trainData <- trainData[,-c(1:6)]
testData <- testData[, -c(1:6)]
dim(trainData)
```

Our new cleaned data set has **53 variables.**


### Cross validation: 60% training 40% testing
```{r}
set.seed(1122)
inTrain <- createDataPartition(trainData$classe,p = 0.6,list=FALSE)
training <- trainData[ inTrain,]
testing <- trainData[-inTrain,]
```


### Training Random Forest
```{r}
library(randomForest)

set.seed(1122)
rf <- randomForest(classe ~ .,data=training)
print(rf)
```

Let's check the accuracy and the sample error with testing data:

```{r}
predictTestingRF <- predict(rf, testing)
confusionMatrix(testing$classe, predictTestingRF)
```

```{r}
accuracyRF <- confusionMatrix(testing$classe, predictTestingRF)$overall[1]
cat("Accuracy: ", accuracyRF)
```
```{r}
seRF <- 1 - confusionMatrix(testing$classe, predictTestingRF)$overall[1]
cat("Out of sample error: ", seRF)
```


### Linear Discriminant Analysis

```{r}
library(caret)

set.seed(1122)
lda <- train(classe ~ ., method = "lda",data=training)
print(lda)
```

Let's check the accuracy with testing data:
```{r}
predictTestingLDA <- predict(lda, testing)
confusionMatrix(testing$classe, predictTestingLDA)
```

```{r}
accuracyLDA <- confusionMatrix(testing$classe, predictTestingLDA)$overall[1]
cat("Accuracy: ", accuracyLDA)
```

```{r}
seLDA <- 1- confusionMatrix(testing$classe, predictTestingLDA)$overall[1]
cat("Out of sample error: ", seLDA)
```


**Random forest method is much more accurate that linear discriminant analysis, and the out of sample error of this method is very small, 0.91%**

```{r}
library(rpart)
library(rpart.plot)
rfModelTree <- rpart(classe ~., data = training, method="class")
prp(rfModelTree,main="Classification Tree")
```

### Run the Prediction Model on the Test Data
Finally, use the random forest prediction model to predict 20 different test cases:
```{r}
predict(rf, testData)
```

### Conclusions

In this project has been appliyed cross-validation in traning data (19622 observations), 70% of the total observations to build a prediction model, and the rest of 30% of the observations to model validation.
Comparing two prediction models, random forest method and linear discriminat analysis, it's concluded that random forest method is much more accurance to predict test cases (accuracy of 99,09% and out of error sample to 0,91%)
The model statistics of this method showed that the sensitivity was in between 97%-99% and the specificity was over 99% for all classes.
In conclusion, the model is well developed to predict the exercise classes during barbell lifts.
