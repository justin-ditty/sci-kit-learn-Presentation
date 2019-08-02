# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 16:16:09 2019

This file shows you how to preprocess data so it can be used with ML estimators 
in the scikit-learn package.

@author: Justin Ditty
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing

data = pd.read_csv("Preprocessing/auto-mpgdata.csv", 
                       header = 1, names = ["mpg", 
                                            "cylinders", 
                                            "displacement", 
                                            "horsepower", 
                                            "weight", 
                                            "acceleration", 
                                            "model_year", 
                                            "origin", 
                                            "car_name"])

# Acceleration is already normally distributed but does not fit the standard 
# normal distribution

# Plot an example of normally distributed data, in this case car acceleration
plt.hist(data["acceleration"])
plt.xlabel("Car Acceleration (mph)")
plt.ylabel("Count")

# The scale function scales data to a standard normal distribution
acceleration_scaled = preprocessing.scale(data["acceleration"])
plt.hist(acceleration_scaled)
plt.xlabel("Scaled Car Acceleration (mph)")
plt.ylabel("Count")

# Plot an example of nonnormaly distributed data, in this case the weight of cars
plt.hist(data["weight"])
plt.xlabel("Car Weight")
plt.ylabel("Count")

# The PowerTrasnformer allows you to map nonnormally distributed data to a 
# normal distribution
pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
weight_transformed = pt.fit_transform(data[["weight"]])
plt.hist(weight_transformed)
plt.xlabel("Car Weight Transformed")
plt.ylabel("Count")

# The train_test_splits methods splits you data into training and testing sets 
# with ease
target = data["mpg"]
features = data["acceleration"]
plt.scatter(features, target)
plt.xlabel("Acceleration (MPH)")
plt.ylabel("Fuel Economy (MPG)")

# The train_test_split module makes it easy to divide the dataset into training 
# and testing subsets
from sklearn.model_selection import train_test_split
features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size = .2, random_state = 0)

# Plot the datasets on the same scatterplot
fig = plt.figure()
axis = fig.add_subplot(111)
axis.scatter(features_train, target_train, color = "blue", label = "Training Set")
axis.scatter(features_test, target_test, color = "red", label = "Testing Set")
plt.xlabel("Acceleration (MPH)")
plt.ylabel("Fuel Economy (MPG)")
plt.legend(loc = "upper left")
plt.show()


