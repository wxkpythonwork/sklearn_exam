#!/usr/bin/env python
# encoding: utf-8

"""
@version: python3.5
@author: wxk
@license: Apache Licence 
@file: pandaslearning.py
@time: 2018/3/7 20:56
"""
import pandas as pd
import numpy as np

#创建索引(不指定就自动创建)
s = pd.Series([1, 7, 13, np.nan, 22])
print(s)
dates = pd.date_range('20180308', periods=6)#自动从20180308+6位
df = pd.DataFrame(np.random.rand(6, 4), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)
print(df['c'])
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20180308'),
                    'C': np.array([3] * 4, dtype='int32'),
                    'D': pd.Categorical(['t', 'e', 's', 't']),
                    'E': 'foo'})
print(df2)
print(df2.index)
print(df2.columns)
print(df2.values)
print(df2.describe)#总结数据
print(df2.T)       #翻转数据transcope
print(df2.sort_index(axis=1, ascending=False))#index(可指定)排序输出
print(df2.sort_values(by='B'))                 #指定列数值排序输出

dates = pd.date_range('20180308',periods=6)
df3 = pd.DataFrame(np.arange(24).reshape(6,4),index=dates,columns=['a', 'b', 'c', 'd'])
print(df3)
print(df3[0:3])                #取行0 1 2，列所有
print(df3['20180308':'20180310'])
print(df3.loc[:,['a','b']])    #使用标签来选择数据-----行所有，列指定(行也可以指定)
df3.iloc[2,2] = 1111           #索引位置修改具体值
print(df3.iloc[[1,3,5],1:3])   #使用位置来选择数据-----行所有，列指定(行也可以指定)
df3.loc['20180309','b'] = 2222 #索引标签修改值
print(df3.ix[:3,['a','b']])    #可以混用位置+标签来选择数据
print(df3[df3.a>8])            #判断指令来选择符合条件的数据（按格式打印出来）

df3.b[df3.a>4]=0               #满足A条件的，修改B在相应位置的值
df3['d']=np.nan                #整列批处理修改
df3['e'] = pd.Series([1,2,3,4,5,6],index=pd.date_range('20180308',periods=6))
print(df3)

df3.iloc[2:5,3] = 21
dp = df3.dropna(axis=0,
           how='any')  #axis=0/1 对行/列操作  any 只要存在就去掉 all 所有行/列为NaN才去掉
print('去掉NaN：\n',dp)
rp = df3.fillna(value=0)#0代替NaN
print('替换NaN：\n',rp)
deci = df3.isnull()
np.any(df3.isnull())==True
print(deci)

#批处理
data = pd.read_csv('C:\\Users\dell\Desktop\V1.csv')
print('读取数据:\n',data)
data.to_pickle('C:\\Users\dell\Desktop\V1.pickle')
#合并
df1 = pd.DataFrame(np.ones((3,6)) * 0, columns=['I','L','O','V','E','U'])
df11 = pd.DataFrame(np.ones((4,6)) * 1, columns=['I','L','O','V','E','U'])
df111 = pd.DataFrame(np.ones((5,6)) * 2, columns=['I','L','O','V','E','U'])
df4 = pd.concat([df1,df11,df111],axis=0,ignore_index=True)#按行，Index忽略
print(df4)

df1 = pd.DataFrame(np.ones((3,6)) * 0, columns=['I','L','O','V','E','U'])
df11 = pd.DataFrame(np.ones((4,6)) * 1, columns=['I','j','k','V','E','U'])
df111 = pd.DataFrame(np.ones((5,6)) * 2, columns=['I','L','O','V','r','U'])
df5 = pd.concat([df1,df11,df111],axis=0,ignore_index=True,join='outer')#按行，outer列相同的合并，没有的另外起一列
df51 = pd.concat([df1,df11,df111],axis=0,ignore_index=True,join='inner')#按行,相同的合并,没有的抛弃
print(df5)

df1 = pd.DataFrame(np.ones((3,4)) * 0,columns=['a','b','c','d'],index=np.arange(3))#columns=list("abcd")
df11 = pd.DataFrame(np.ones((3,4)) * 1,columns=['a','b','e','f'],index=np.arange(2,5,1))
df6 = pd.concat([df1,df11],axis=1,join_axes=[df1.index])#按照df1的index来合并，有就显示
print(df6)

s = pd.Series([1,2,3,4],index=['a','b','c','d'])        #append 只有纵向合并 没有横向合并
df6 = pd.DataFrame(np.ones((3,4)) * 1,columns=['a','b','c','d'],index=np.arange(3))
df66 = df6.append(df1,ignore_index=True)
print('df66:\n',df66)
df7 = df1.append(s,ignore_index=True)
print('df7:\n',df7)