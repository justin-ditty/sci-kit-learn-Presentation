# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 21:00:12 2019

This code contains an example of how to use the sci-kit learn package  to fit, 
train and evaluate a linear regression model that 

For this example we will model an automible's fuel efficiency based on a number
of different quantitative factors about the car.

@author: Justin Ditty
"""

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt

# Use pandas read_csv method to import the dataset
autosmpg = pd.read_csv("Regression/auto-mpgdata.csv", 
                       header = 1, names = ["mpg", 
                                            "cylinders", 
                                            "displacement", 
                                            "horsepower", 
                                            "weight", 
                                            "acceleration", 
                                            "model_year", 
                                            "origin", 
                                            "car_name"])
# Examine the head of the dataset#
autosmpg.head()

# Pandas includes many types of plots that are useful for evaluating the relationships between target and feature variables 
# visually such as a scatterplot matrix
from pandas.plotting import scatter_matrix
scatter_matrix(autosmpg, alpha=0.2, figsize=(9, 9), diagonal='kde')

# Promising features are displacement, weight, acceleration, and year
# Transforming the variables can increase the accuracy of the regression model.
autosmpg["recip_displacement"] = 1/autosmpg["displacement"] 
autosmpg["recip_weight"] = 1/autosmpg["weight"] 
autosmpg["log_acceleration"] = np.log(autosmpg["acceleration"])

# Examine the relationship between the transformed variables ands the car's gas
# milage.
scatter_matrix(autosmpg, alpha=0.2, figsize=(9, 9), diagonal='kde')

# Select the columns that contain data for the predictor variables
# In this case they are the recorcal of the cars weight, 
# the log of the car's acceleration and the car's model year.
feature_cols = ["recip_weight", "sqrt_acceleration", "model_year"]
autos_features = autosmpg[feature_cols]

# Select the columns containing the target variable, in this case miles per gallon
autos_target = autosmpg['mpg']

# The train_test_split module makes it easy to divide the dataset into training and testing subsets
from sklearn.model_selection import train_test_split

#Split the data into training and testing sets
autos_features_train, autos_features_test, autos_target_train, autos_target_test = train_test_split(
        autos_features, autos_target, test_size = .2, random_state = 0)

# Regression results are stored in a linear regression object
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()

# Fit the linear regression model using the taining sets
# Using the fit method
lin_reg.fit(autos_features_train, autos_target_train)

# The accuracy of the model can be greatly improved by using k-fold cross 
# validation n which the model is fit on k different subsets of the training 
# data before attempting to predict the testing set
from sklearn.model_selection import cross_val_score
lin_reg_score = cross_val_score(lin_reg, autos_features_train, 
                                autos_target_train, cv=20)
np.mean(lin_reg_score)

#Make predicitons on the testing set
autos_predict = lin_reg.predict(autos_features_test)

# Evaluate the fit
from sklearn.metrics import mean_squared_error, r2_score
print('Coefficient: \n ', lin_reg.coef_)
print("MSE: %.2f" % mean_squared_error(autos_target_test, autos_predict))
print("r^2: %.2f" % r2_score(autos_target_test, autos_predict))

