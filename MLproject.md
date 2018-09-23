setwd("C:/Users/m.hernandez.moreno/Desktop/Curso Data Science/08. Practical Machine Learning/Project")

### Getting and cleaning data

# Loading the training and testng data set
trainData <- read.csv("pml-training.csv")
testData <- read.csv("pml-testing.csv")

dim(trainData)
dim(testData)

#Of those 160 variables, select ones with high (over 95%) missing data and exclude them from the analysis:
trainData[trainData==""] <- NA
NArate <- apply(trainData, 2, function(x) sum(is.na(x)))/nrow(trainData)
trainData <- trainData[!(NArate>0.95)]

testData[testData==""] <- NA
NArate <- apply(testData, 2, function(x) sum(is.na(x)))/nrow(testData)
testData <- testData[!(NArate>0.95)]

dim(trainData)

#Remove variables have near zero variance and remove them from the data:
library(caret)
nzv <- nearZeroVar(trainData, saveMetrics = TRUE)
trainData <- trainData[,nzv$nzv == "FALSE"]
trainData$classe <- as.factor(trainData$classe)

nzv <- nearZeroVar(testData, saveMetrics = TRUE)
testData <- testData[, nzv$nzv == "FALSE"]

dim(trainData)
colnames(trainData)

#Delete from variable 1 to variable 6 because they can't be predictors:
trainData <- trainData[,-c(1:6)]
testData <- testData[, -c(1:6)]
dim(trainData)


### Cross validation: 60% training 40% testing
set.seed(1122)
inTrain <- createDataPartition(trainData$classe,p = 0.6,list=FALSE)
training <- trainData[ inTrain,]
testing <- trainData[-inTrain,]


### Training Random Forest

library(randomForest)

set.seed(1122)
rf <- randomForest(classe ~ .,data=training)
print(rf)

#Let's check the accuracy and the sample error with testing data:

predictTestingRF <- predict(rf, testing)
confusionMatrix(testing$classe, predictTestingRF)

accuracyRF <- confusionMatrix(testing$classe, predictTestingRF)$overall[1]
cat("Accuracy: ", accuracyRF)

seRF <- 1 - confusionMatrix(testing$classe, predictTestingRF)$overall[1]
cat("Out of sample error: ", seRF)


### Linear Discriminant Analysis

library(caret)

set.seed(1122)
lda <- train(classe ~ ., method = "lda",data=training)
print(lda)

#Let's check the accuracy with testing data:
predictTestingLDA <- predict(lda, testing)
confusionMatrix(testing$classe, predictTestingLDA)

accuracyLDA <- confusionMatrix(testing$classe, predictTestingLDA)$overall[1]
cat("Accuracy: ", accuracyLDA)

seLDA <- 1- confusionMatrix(testing$classe, predictTestingLDA)$overall[1]
cat("Out of sample error: ", seLDA)
# Random forest method is much more accurate that linear discriminant analysis,
# and the out of sample error of this method is very small, 0.91%**


library(rpart)
library(rpart.plot)
png("lassification_Tree.png", width=480, height=480)
rfModelTree <- rpart(classe ~., data = training, method="class")
prp(rfModelTree,main="Classification Tree")
dev.off()

### Run the Prediction Model on the Test Data
#Finally, use the random forest prediction model to predict 20 different test cases:

predict(rf, testData)
