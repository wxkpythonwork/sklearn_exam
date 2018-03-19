#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.5
@author: wxk
@license: Apache Licence 
@file: classifier.py
@time: 2018/3/7 16:26
"""
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_digits
#三个交叉验证
from sklearn.learning_curve import learning_curve
from sklearn.learning_curve import validation_curve
from sklearn.cross_validation import cross_val_score

# from sklearn.cross_validation import train_test_split #将数据划分，留一法？

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import preprocessing  #normalization处理模块
from sklearn.datasets.samples_generator import make_classification
from sklearn.svm import SVC
import pickle

# iris = datasets.load_iris()
# X_data = iris.data
# y_data = iris.target
# X_train,X_test,y_train,y_test = train_test_split(X_data,y_data,test_size=0.3)
# K_range = range(1,31)
# k_scores = []
# for k in K_range:
#     knn = KNeighborsClassifier(n_neighbors=k)
#
#     knn.fit(X_train,y_train)
# # score = knn.score(X_test, y_test)
#     #score = cross_val_score(knn,X_data,y_data,cv=10,scoring='accuracy')
#     score = cross_val_score(knn, X_data, y_data, cv=10, scoring='mean_squared_error')
#     k_scores.append(score.mean())
# plt.plot(K_range,k_scores,'g-o')
# plt.xlabel('K-values')
# plt.ylabel('scores-value')
# plt.title('Scores-K')
# plt.legend()
# plt.show()

# boston = datasets.load_boston()
# data_X = boston.data
# data_y = boston.target
# X_train,X_test,y_train,y_test = train_test_split(data_X,data_y,test_size=0.3)
# model =  LinearRegression()
# model.fit(X_train,y_train)
# result = model.predict(X_test)
# print(result[:4,])
# print(y_test[:4,])
# print(model.coef_)#y = 0.1x+0.3 输出0.1
# print(model.intercept_)#输出0.3
# print(model.get_params())
# print(model.score(data_X,data_y))
# 可视化数据
# X,y = datasets.make_regression(n_samples=500,n_features=1,n_targets=1,noise=10)
# plt.scatter(X,y)
# plt.show()

# a = np.array([[10,2.5,3.6],
#               [-100,5,-8],
#               [120,20,40]],dtype=np.float64)
# print(preprocessing.scale(a))
X,y = make_classification(n_samples=300,n_features=2,n_informative=2,n_redundant=0,random_state=22,n_clusters_per_class=1,scale=100)
# plt.scatter(X[:,0],X[:,1],y)
# plt.show()
# X = preprocessing.scale(X)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# model = SVC()
# model.fit(X_train, y_train)
# print(model.score(X_test, y_test))

# digits = load_digits()
# x = digits.data
# y = digits.target
# train_sizes, train_loss, test_loss = learning_curve(SVC(gamma=0.001),x,y,cv=10,scoring='mean_squared_error',train_sizes=[0.1,0.25,0.5,0.75,1])#在10%，25%...样本取点
# train_loss_mean = -np.mean(train_loss,axis=1)
# test_loss_mean = -np.mean(test_loss,axis=1)
# plt.plot(train_sizes,train_loss_mean,'*-',color='g',label='Training')
# plt.plot(train_sizes,test_loss_mean,'-o',color='r',label='Testing')
# plt.xlabel('train_sizes')
# plt.ylabel('loss')
# plt.legend(loc='best')
# plt.show()


# digits = load_digits()
# x = digits.data
# y = digits.target
# param_range = np.logspace(-6, -2.3, 5)#参数测试集
# # train_loss, test_loss = validation_curve(SVC(),x,y,param_name='gamma',param_range = param_range,cv=10,scoring='mean_squared_error')
# train_loss, test_loss = validation_curve(SVC(),x,y,cv=10,param_name='gamma',param_range=param_range,scoring='mean_squared_error')
# train_loss_mean = -np.mean(train_loss,axis=1)
# test_loss_mean = -np.mean(test_loss,axis=1)
# plt.plot(param_range,train_loss_mean,'o-',color='g', label='train')
# plt.plot(param_range,test_loss_mean,'*-',color='r', label='cross-validation')
# plt.xlabel('gamma')
# plt.ylabel('loss')
# plt.legend(loc='best')
# plt.show()


iris = datasets.load_iris()
X_data = iris.data
y_data = iris.target
clf = SVC()
clf.fit(X_data,y_data)
#method 1

#save模型
# with open('save/clf.pickle','wb') as f:
#     pickle.dump(clf,f)
# #read模型
# with open('save/clf.pickle','rb') as f:
#     pickle.load(f)
#method 2 更快！！！

from sklearn.externals import joblib
#save
joblib.dump(clf,'save/clf.pkl')
#reload
clfload = joblib.load('save/clf.pkl')
print(clfload.predict(X_data[0:1]))