# Machine learning project: How well do you barbell lifts?

This R Markdown describes the steps following to create a prediction model. You can find the R code in the *MLproject.html* and *MLproject.Rmd*

## Summary
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways:
        
* Class A: exactly according to the specification

* Class B: throwing the elbows to the front

* Class C: lifting the dumbbell only halfway

* Class D: lowering the dumbbell only halfway

* Class E: throwing the hips to the front

Class A corresponds to the specified execution of the exercise, while the other 4 classes correspond to common mistakes.

## Getting and Cleaning Data
After loading training data set (19622 observations and 160 variables) and testing data set (20 observations and 160 variables), some variables were removed:

* Variables with high (over 95%) missing data

* Variables which have near zero variance 

* Variables which have nothing to do with making the predictions

The new cleaned data set has **53 variables.**

## Cross validation
Making 60% training and 40% testing cross-validation, two prediction models were applied:
1. Random Forest Method

2. Linear Discriminat Analysis

Comparing **accuracy** for testing set, random forest method is better, 99.09% versus 69.1%. The **out of sample error** was 0.91% and 30.9% respectively.

## Use prediction model on the test data
Applying random forest method to predict 20 different test cases, the result was:

 1: B
 
 2: A
 
 3: B
 
 4: A
 
 5: A
 
 6: E
 
 7: D
 
 8: B
 
 9: A
 
 10: A
 
 11: B
 
 12: C
 
 13: B
 
 14: A
 
 15: E
 
 16: E
 
 17: A
 
 18: B
 
 19: B
 
 20: B
 
 ## Conclusions
In this project has been appliyed cross-validation in traning data (19622 observations), 70% of the total observations to build a prediction model, and the rest of 30% of the observations to model validation.
Comparing two prediction models, random forest method and linear discriminat analysis, it's concluded that random forest method is much more accurance to predict test cases (accuracy of 99,09% and out of error sample to 0,91%)
The model statistics of this method showed that the sensitivity was in between 97%-99% and the specificity was over 99% for all classes.

In conclusion, the model is well developed to predict the exercise classes during barbell lifts.

