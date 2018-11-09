# @Time    : 2018/11/8 13:00
# @Author  : Yanlin Wang
# @Email   : wangyl_a@163.com
# @File    : function_detail.py

from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imp
from sklearn.linear_model import LinearRegression
# 误差评估
from sklearn import metrics





def error_caculate(primary_value, predict_value):
    return 1 - (np.std(np.array(predict_value[1:]) - np.array(primary_value[:-1]) - (
        np.array(primary_value[1:] - np.array(primary_value[:-1]))), ddof=1) / np.std(
        np.array(primary_value[1:]) - np.array(primary_value[:-1]), ddof=1))


def error_analysis_plot(primary_data, predicted_data, plot_open = False):
    # 误差分析, 平均绝对误差(MAE)
    print('误差:', error_caculate(primary_data, predicted_data))
    print('MAE', metrics.mean_absolute_error(primary_data, predicted_data))
    # # print('MSE', metrics.mean_squared_error(df2.T12A041A, df2.predict_data))
    print('RMSE', np.sqrt(metrics.mean_squared_error(primary_data, predicted_data)))

    # 绘图
    if plot_open:
        plt.figure()
        plt.plot(primary_data, color='red', linewidth=2, label='primary_data')
        plt.plot(predicted_data,color='blue', linewidth=2, label='predicted_data')
        plt.legend(loc='best')
if __name__ == '__main__':

    start = clock()
    end = clock()
    print('time: {:.8f}s'.format(end - start))
