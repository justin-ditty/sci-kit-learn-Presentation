# -*- coding: utf-8 -*-
"""
Created on Fri Jul 19 04:39:53 2019

In this example I will show how to use classifcation methods from the sci-kit 
learn paskage to detect the precence of breast cancer using decision trees.

@author: Justin Ditty
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

#import numpy as np
#import matplotlib.pyplot as plt

#Classification Example
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.model_selection import train_test_splits

cancer = pd.read_csv("C:/Users/justi/OneDrive/Documents/TDAI sci-kit learn final/Classification/breast-cancerdata.csv", 
                     names = ["class", 
                              "age", 
                              "menopause", 
                              "tumor-size", 
                              "inv-nodes", 
                              "node-caps", 
                              "def_malig", 
                              "breast", 
                              "breast-quad", 
                              "irradiat"])

#Check the head of the data set
cancer.head()

# Select feature and target columns
feature_cols = ["age","menopause", "tumor-size", "inv-nodes", "node-caps", 
                "def_malig", "breast", "breast-quad", "irradiat"]

cancer_target = cancer["class"]
cancer_features = cancer[feature_cols]

#Use an ordinal encoder to preprocess the data
encoder = OrdinalEncoder()

#Transform all of the categorical features into numerical values
cancer_features_encoded = encoder.fit_transform(cancer_features, cancer_target)

#Split into training and testing sets
cancer_features_train, cancer_features_test, cancer_target_train, cancer_target_test = train_test_split(cancer_features_encoded, cancer_target)

#create the cancer decision tree object
#arguements are default
cancer_tree = DecisionTreeClassifier(criterion = "gini", 
                              splitter="best",
                              max_depth=None, 
                              min_samples_split=2,
                              min_samples_leaf=1, 
                              min_weight_fraction_leaf=0,
                              max_features=None, 
                              random_state=None, 
                              max_leaf_nodes=None, 
                              min_impurity_decrease=1e-7,
                              class_weight=None)

#Train the decision tree on the training data set using k-fold cross validation
cancer_tree = cancer_tree.fit(cancer_features_train, cancer_target_train)

cancer_scores = cross_val_score(cancer_tree, cancer_features_train, cancer_target_train, cv=20)
np.mean(cancer_scores)
#Predict breast cancer occurence in the testing dataset
cancer_tree_predict = cancer_tree.predict(cancer_features_test)

# Random forest
rf = RandomForestClassifier()

# Generate the forest of decision trees
cancer_rf = rf.fit(cancer_features_train, cancer_target_train)
cancer_scores_rf = cross_val_score(cancer_rf, cancer_features_train, cancer_target_train, cv=20)
np.mean(cancer_scores_rf)

# Predict the testing set
cancer_prediction_rf = cancer_rf.predict(cancer_features_test)

# Evaluate Model Accuracy
correctly_labelled_points_tree = (cancer_target_test == cancer_tree_predict).sum()
accuracy_tree = correctly_labelled_points_tree/len(cancer_target_test)*100
print("Decision Tree Accuracy: %.2f" % accuracy_tree)

correctly_labelled_points_rf = (cancer_target_test == cancer_prediction_rf).sum()
accuracy_rf = correctly_labelled_points_rf/len(cancer_target_test)*100
print("Random Forest Accuracy: %.2f" % accuracy_rf)