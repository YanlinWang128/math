# @Time    : 2018/11/7 21:38
# @Author  : Yanlin Wang
# @Email   : wangyl_a@163.com
# @File    : Bayesian Ridge Regression.py

from sklearn import linear_model, metrics
from sklearn import svm
from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import imp


def error_caculate(primary_value, predict_value):
    return 1 - (np.std(np.array(predict_value[1:]) - np.array(primary_value[:-1]) - (
        np.array(primary_value[1:] - np.array(primary_value[:-1]))), ddof=1) / np.std(
        np.array(primary_value[1:]) - np.array(primary_value[:-1]), ddof=1))


def linear_model_analysis(flag=1):
    df1 = pd.read_csv('D:/08_35.csv')  # .dropna()
    df2 = pd.read_csv('D:/09.csv')  # .dropna()

    x = df1[['LAECF411', 'u2', 'u3', 'TOTFUELF']]
    y = df1.T12A041A

    if flag is 1:
        print('方法: 普通最小二乘线性回归')
        linreg = linear_model.LinearRegression().fit(x, y)  # 线性回归建模
    elif flag is 2:
        print('Bayesian Ridge Regression')
        linreg = linear_model.BayesianRidge().fit(x, y)  # Bayesian Ridge Regression 回归
    elif flag is 3:
        print('用最小角回归法求解Lasso回归')
        linreg = linear_model.LassoLars(alpha=.1).fit(x, y)
    elif flag is 4:
        print('逻辑回归')
        linreg = linear_model.LogisticRegression().fit(x, y.astype('int'))
    elif flag is 5:
        print('支持向量机')
        linreg = svm.SVC().fit(x, y.astype('int'))
    # 线性回归模型预测, 输入x, 输出y'
    df2['predict_data'] = linreg.predict(df2[['LAECF411', 'u2', 'u3', 'TOTFUELF']])
    # print(df2['predict_data'])
    if flag < 5:
        print(linreg.coef_, linreg.intercept_) #

    print('第二种误差: ', error_caculate(df2.T12A041A, df2.predict_data))

    # 误差分析, 平均绝对误差(MAE)
    print('MAE', metrics.mean_absolute_error(df2.T12A041A, df2.predict_data))
    # print('MSE', metrics.mean_squared_error(df2.T12A041A, df2.predict_data))
    print('RMSE', np.sqrt(metrics.mean_squared_error(df2.T12A041A, df2.predict_data)))
    print('------')
    plt.figure()
    df2.T12A041A.plot(color='red', linewidth=2, label='primary_data')
    df2.predict_data.plot(color='blue', linewidth=2, label='predicted_data')
    plt.legend(loc='best')
    # plt.show()


if __name__ == '__main__':
    start = clock()
    linear_model_analysis(1)
    linear_model_analysis(2)
    linear_model_analysis(3)
    linear_model_analysis(4)
    # linear_model_analysis(5)

    plt.show()
    end = clock()
    print('time: {:.8f}s'.format(end - start))
