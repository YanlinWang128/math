# @Time    : 2018/11/7 14:52
# @Author  : Yanlin Wang
# @Email   : wangyl_a@163.com
# @File    : 2. 回归分析_线性回归.py

from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imp

start = clock()

df = pd.DataFrame({'data1': [1.5, 2.8, 4.5, 7.5, 10.5, 13.5, 15.1, 16.5, 19.5, 22.5, 24.5, 26.5],
                   'data2': [7.0, 5.5, 4.6, 3.6, 2.9, 2.7, 2.5, 2.4, 2.2, 2.1, 1.9, 1.8]})
x = df[['data1']]
y = df.data2


#  y = coef_ * Xi + intercept_
# print(x, y)
from sklearn.linear_model import LinearRegression

# 线性回归建模
linreg = LinearRegression().fit(x, y)

# 线性回归模型预测, 输入x, 输出y'
df['predict_data'] = linreg.predict(df[['data2']])
print(df['predict_data'])
print(linreg.coef_, linreg.intercept_)
# print(df)
end = clock()
print('time: {:.8f}s'.format(end - start))

"""
Scikit-learn是个简单高效的数据分析工具，它其中封装了大量的机器学习算法，内置了大量的公开数据集，并且拥有完善的文档。
-- 线性回归方法,参数
f(x) = wx + b
LinearRegression(fit_intercept=True,normalize=False,copy_X=True,n_jobs=1)

sklearn.linear_model.LinearRegression参数

sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)

    fit_intercept ：（截距）默认为True，可选False 
    normalize：（标准化） 默认为True，可选False 
    copy_X：（复制X数据）默认为True，可选False。如果选False会覆盖原数 
    n_jobs：（计算性能）默认为1，可选int，工作使用的数量计算。


"""
