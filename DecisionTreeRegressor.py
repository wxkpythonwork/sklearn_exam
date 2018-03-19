#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.5
@author: wxk
@license: Apache Licence 
@file: DecisionTreeRegressor.py
@time: 2018/3/5 10:48
"""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
from sklearn import svm
from sklearn import neighbors
from sklearn import ensemble
import matplotlib.pyplot as plt

def f(x1, x2):
    y = 0.5 * np.sin(x1) + 0.5 * np.cos(x2)  + 0.1 * x1 + 3
    return y

def load_data():
    x1_train = np.linspace(0,50,500)
    x2_train = np.linspace(-10,10,500)
    data_train = np.array([[x1,x2,f(x1,x2) + (np.random.random(1)-0.5)] for x1,x2 in zip(x1_train, x2_train)])
    x1_test = np.linspace(0,50,100)+ 0.5 * np.random.random(100)
    x2_test = np.linspace(-10,10,100) + 0.02 * np.random.random(100)
    data_test = np.array([[x1,x2,f(x1,x2)] for x1,x2 in zip(x1_test, x2_test)])
    return data_train, data_test

def DiffenertWay(clf):
    clf.fit(x_train,y_train)
    result = clf.predict(x_test)
    score = clf.score(x_test,y_test)
    print(result)
    print('Score:%f' %score)
    plt.figure()
    plt.plot(np.arange(len(result)),y_test,'go-',label='True value')
    plt.plot(np.arange(len(result)),result,'ro-',label='Predict value')
    plt.title('Score:%f' %score)
    plt.legend()
    plt.show()

train, test = load_data()
x_train = train[:,:2]
y_train = train[:,2]
x_test = test[:,:2]
y_test = test[:,2]


clf = DecisionTreeRegressor(criterion='mse', max_depth=None, max_features=None,
                      max_leaf_nodes=None, min_samples_leaf=1, min_samples_split=2,
                      min_weight_fraction_leaf=0.0, presort=False, random_state=None,
                      splitter='best')
DiffenertWay(clf)

clf = linear_model.LinearRegression()
DiffenertWay(clf)

clf = svm.SVR()
DiffenertWay(clf)

clf = neighbors.KNeighborsRegressor()
DiffenertWay(clf)

clf = ensemble.RandomForestRegressor()
DiffenertWay(clf)

clf = ensemble.AdaBoostRegressor()
DiffenertWay(clf)

clf = ensemble.GradientBoostingRegressor()
DiffenertWay(clf)