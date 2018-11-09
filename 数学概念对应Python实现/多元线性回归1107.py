# @Time    : 2018/11/7 15:49
# @Author  : Yanlin Wang
# @Email   : wangyl_a@163.com
# @File    : 多元线性回归1107.py

from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imp

start = clock()

df1 = pd.read_csv('D:/08_35.csv')  # .dropna()
df2 = pd.read_csv('D:/09.csv')  # .dropna()

# 检测是够有缺失值
# print(df1.isnull().sum())
# print(df2.isnull().sum())
x = df1[['LAECF411', 'u2', 'u3', 'TOTFUELF']]
y = df1.T12A041A
#  y = coef_ * Xi + intercept_
# print(x, y)
from sklearn.linear_model import LinearRegression
# 误差评估
from sklearn import metrics

# 线性回归建模
linreg = LinearRegression().fit(x, y)

# 线性回归模型预测, 输入x, 输出y'
df2['predict_data'] = linreg.predict(df2[['LAECF411', 'u2', 'u3', 'TOTFUELF']])
# print(df2['predict_data'])
print(linreg.coef_, linreg.intercept_)


def error_caculate(primary_value, predict_value):
    return 1 - (np.std(np.array(predict_value[1:]) - np.array(primary_value[:-1]) - (
        np.array(primary_value[1:] - np.array(primary_value[:-1]))), ddof=1) / np.std(
        np.array(primary_value[1:]) - np.array(primary_value[:-1]), ddof=1))


print(error_caculate(df2.T12A041A, df2.predict_data))


# 误差分析, 平均绝对误差(MAE)
"""
显著性目标检测算法常用的评价指标有：
平均绝对误差(Mean Absolute Error, MAE)
PR曲线(Precision-Recall curves)
F度量值(F-measure)
"""
print('MAE', metrics.mean_absolute_error(df2.T12A041A, df2.predict_data))
# print('MSE', metrics.mean_squared_error(df2.T12A041A, df2.predict_data))
print('RMSE', np.sqrt(metrics.mean_squared_error(df2.T12A041A, df2.predict_data)))
df2.T12A041A.plot(color='red', linewidth=2, label='primary_data')
df2.predict_data.plot(color='blue', linewidth=2, label='predicted_data')
plt.legend(loc='best')
# plt.show()
end = clock()
print('time: {:.8f}s'.format(end - start))
