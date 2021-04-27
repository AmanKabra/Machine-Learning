# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 17:28:24 2021

@author: Aman Kabra
"""

from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784')

from sklearn.model_selection import train_test_split
train_img, test_img, train_lbl, test_lbl = train_test_split( mnist.data, 
                                                            mnist.target, 
                                                            test_size=1/7.0, 
                                                            random_state=0)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
# Fit on training set only.
scaler.fit(train_img)
# Apply transform to both the training set and the test set.
train_img = scaler.transform(train_img)
test_img = scaler.transform(test_img)

from sklearn.decomposition import PCA
# Make an instance of the Model
pca = PCA(.95)

pca.fit(train_img)

train_img = pca.transform(train_img)
test_img = pca.transform(test_img)



from sklearn.linear_model import LogisticRegression

# all parameters not specified are set to their defaults
# default solver is incredibly slow which is why it was changed to 'lbfgs'
logisticRegr = LogisticRegression(solver = 'lbfgs')

!git init

!git stage


















