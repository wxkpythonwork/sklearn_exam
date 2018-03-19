#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.5
@author: wxk
@license: Apache Licence 
@file: diabets.py
@time: 2018/3/8 17:05
"""

import pandas as pd
import numpy as np
import time
from sklearn.metrics import mean_squared_error
import xgboost
from sklearn import grid_search
from sklearn.model_selection import train_test_split #将数据划分，留一法？
from sklearn import preprocessing
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.decomposition import PCA

from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import make_pipeline

def mse(y_true,y_pre):
    m = len(y_true)
    sum = 0.0
    for i , j in zip(y_true, y_pre):
        t = j - i
        sum += t * t
    return sum / (2 * m)

def count_info(data):#数据信息
    print(data.head(5))
    round(data.describe())
    print(data.shape)
    blood = data.loc[:,'性别']
    bloodinfo = blood.describe()
    print(bloodinfo)

def clean_data(data,test_A):

    data = data[data['血糖'] < 20]
    data = data[data['年龄'] > 20]
    label = data['血糖']
    del data['血糖']
    data = data.fillna(data.mean(axis=0))
    test_A = test_A.fillna(test_A.mean(axis=0))
    #合并数据
    all_data = pd.concat([data,test_A],axis=0,join='outer',ignore_index=True)
    all_info = all_data.drop(['id','体检日期','乙肝表面抗原', '乙肝表面抗体', '乙肝e抗原', '乙肝e抗体', '乙肝核心抗体'], axis=1)
    all_info = all_info.replace({'男','女','??'},{1,-1,0})
    all_info['TG_UA'] = all_info['总胆固醇'] / all_info['尿酸']
    all_info['ALT_AST'] = all_info['*丙氨酸氨基转换酶'] / all_info['*天门冬氨酸氨基转换酶']
    #分割数据
    train_data = all_info.iloc[:data.shape[0],:]
    test_data = all_info.iloc[data.shape[0]:,:]

    scaler = preprocessing.StandardScaler().fit(train_data)
    train_data = scaler.transform(train_data)  #标准偏差scaler.scale_, scaler.mean_scaler.transform()
    scaler1 = preprocessing.StandardScaler().fit(test_data)
    test_data = scaler1.transform(test_data)
    # test = scaler.transform(test_A)
    return train_data,test_data,label



if __name__ == '__main__':
    start = time.time()
    data = pd.read_csv('d_train_20180102.csv', encoding='gb2312')
    count_info(data)
    print(data.shape)
    test_A = pd.read_csv('d_test_A_20180102.csv', encoding='gb2312')
    print(test_A.shape)
    train_data,pre_data,label = clean_data(data,test_A)
    # pca = PCA(n_components=15)
    # pca.fit(train_data)
    x_train, x_test, y_train, y_test = train_test_split(train_data, label, test_size=0.2, random_state=2018)
    clf = xgboost()
    clf.fit(x_train, y_train)
    #error = mean_squared_error(y_test,clf.predict(x_test))
    error = mse(y_test,clf.predict(x_test))
    print(error)
    end = time.time()
    print('用时{}秒'.format(end - start))





