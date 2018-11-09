# @Time    : 2018/11/8 14:49
# @Author  : Yanlin Wang
# @Email   : wangyl_a@163.com
# @File    : 多元回归分析_真实数据_1108.py

from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imp
from sklearn.linear_model import LinearRegression
from sklearn import metrics  # 误差评估
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from function_detail import *


def analysis1108(delay, plot_open=False):
    input_path = r'D:/all08_.csv'
    d1 = d2 = d3 = d4 = delay

    na, nb1, nb2, nb3, nb4 = 5, 10, 5, 5, 5
    df2 = pd.read_csv(input_path, header=0)

    y_output = df2['T12A041A'].tolist()

    u1_difference = df2['u1_difference'].tolist()
    u2_difference = df2['u2_difference'].tolist()
    u3_difference = df2['u3_difference'].tolist()
    u4_difference = df2['u4_difference'].tolist()

    # start_item = max([na, d1 + nb1, d2 + nb2, d3 + nb3, d4 + nb4]) + 3
    # 开始,结束索引
    start_item = max([na, 1 + nb1, 1 + nb2, 1 + nb3, 1 + nb4]) + 2
    end_item = len(df2.index) - 1
    max_length = end_item - start_item + 1
    columns = ['_y_k_' + str(k) for k in range(1, na + 1)] + ['deltaU1_k_1_' + str(k) for k in range(0, nb1 + 1)] + [
        'deltaU2_k_1_' + str(k) for k in range(0, nb2 + 1)] + ['deltaU3_k_1_' + str(k) for k in range(0, nb3 + 1)] + [
                  'deltaU4_k_1_' + str(k) for k in range(0, nb4 + 1)]
    dfnew = pd.DataFrame(columns=columns)

    def data_create():

        for i in range(1, na + 1):
            dfnew['_y_k_' + str(i)] = -1 * np.array(y_output[start_item - i - 1:][:max_length])
        for i in range(0, nb1 + 1):
            dfnew['deltaU1_k_1_' + str(i)] = u1_difference[start_item - i - 1 - 2:][:max_length]
        for i in range(0, nb2 + 1):
            dfnew['deltaU2_k_1_' + str(i)] = u2_difference[start_item - i - 1 - 2:][:max_length]
        for i in range(0, nb3 + 1):
            dfnew['deltaU3_k_1_' + str(i)] = u3_difference[start_item - i - 1 - 2:][:max_length]
        for i in range(0, nb4 + 1):
            dfnew['deltaU4_k_1_' + str(i)] = u4_difference[start_item - i - 1 - 2:][:max_length]

            # map(lambda i: dfnew['_y_k_' + str(i)] = - y_output[start_item:], range(1, na + 1))

    data_create()
    # dfnew.to_csv('a.csv', index=False)
    x = dfnew[columns]
    # print(len(x))
    y = y_output[start_item - 1:][:max_length]
    # print(len(y))
    linreg = LinearRegression().fit(x, y)
    # print(linreg.coef_, linreg.intercept_)

    dfnew['predict_data'] = linreg.predict(dfnew[columns])
    # print(len(dfnew))
    error_analysis_plot(y, dfnew['predict_data'], plot_open)
    # print(dfnew.head())

    # print(dfnew.columns)
    # for i in range(start_item, end_item):
    #     pass


if __name__ == '__main__':
    start = clock()

    analysis1108(10, True)
    plt.show()
    end = clock()
    print('time: {:.8f}s'.format(end - start))
