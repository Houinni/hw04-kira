[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/wbf5VqoP)
# CSC/SDS293: Homework 04

## Description
The purpose of this homework is perform Linear Model Reularization Methods and to use Cross Validation when appropriate. 

## Instructions
  

### Part 1: College 

For this part of the assignment you will work with the College dataset included in the ISL library, and will predict the number of applications received using the other variables in the College dataset. 

1. Split your data into a training and test set. Use a seed of 1 and take a random sample of 512 to be your training set.  

1. Fit a linear model using least squares on the training set, and report the test error obtained.

1. Fit a ridge regression model on the training set, with lambda chosen
by cross-validation. Report the test error obtained.

1. Fit a lasso model on the training set, with lambda chosen by cross-validation. Report the test error obtained, along with the number of non-zero coefficient estimates.

1. Fit a PCR model on the training set, with M chosen by cross-validation.
Report the test error obtained, along with the value of M selected by cross-validation.

1. Fit a PLS model on the training set, with M chosen by cross-validation. Report the test error obtained, along with the value of M selected by cross-validation.

1. Comment on the results obtained. 
    * How accurately can we predict the number of college applications received? 
    * Is there much difference among the test errors resulting from these five approaches? 

### Part 2: Simulation

We have seen that as the number of features used in a model increases, the training error will necessarily decrease, but the test error may not. We will now explore this in a simulated data set.
 
1. Generate a data set with p = 20 features, n = 1,000 observations, and an associated quantitative response vector generated according to the model
Y = X\*beta + epsilon, where beta has some elements that are exactly equal to zero. 

1. Split your data set into a training set containing 100 observations and a test set containing 900 observations.

1. Perform best subset selection on the training set, and plot the training set MSE associated with the best model of each size.

1. Plot the test set MSE associated with the best model of each size.

1. For which model size does the test set MSE take on its minimum value? 
    * If it takes on its minimum value for a model containing only an intercept or a model containing all of the features, then play around with the way that you are generating the data until you come up with a scenario in which the test set MSE is minimized for an intermediate model size.  

## Submission and Grading
Please see the assignment overview from the course website for submission instrucitons and rubric. 
