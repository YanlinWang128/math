# @Time    : 2018/11/8 10:10
# @Author  : Yanlin Wang
# @Email   : wangyl_a@163.com
# @File    : functions.py


from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import imp
from sklearn.linear_model import LinearRegression
from sklearn import metrics# 误差评估
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


from function_detail import *
#  y = coef_ * Xi + intercept_
def linear_regression(x, y):
    linreg = LinearRegression().fit(x, y)
    print(linreg.coef_, linreg.intercept_)

    # 线性回归模型预测, 输入x, 输出y'
    # df2['predict_data'] = linreg.predict(df2[['LAECF411', 'u2', 'u3', 'TOTFUELF']])

    # print(df2['predict_data'])



def analysis1108(delay, plot_open=False):
    input_path = r'C:/Users/Frank/Desktop/tongliu/mat_data_1102.csv'
    d1 = d2 = d3 = d4 = delay

    na, nb1, nb2, nb3, nb4 = 5, 10, 5, 5, 5
    # phi_(i)的长度
    matrix_length = na + nb1 + nb2 + nb3 + nb4 + 4

    # 初始值
    p = np.eye(matrix_length) * (10 ** 10)  # 每个文件重置一次
    theta = np.zeros((matrix_length, 1))

    df2 = pd.read_csv(input_path, header=0)

    # print(df2.columns.values.tolist())
    # theta_first_value = [0]
    y_output = df2['y'].tolist()
    difference_length = len(df2.index) - 1
    # print(difference_length)

    u1_difference = df2['u_difference'].tolist()
    u2_difference = [0] * difference_length
    u3_difference = [0] * difference_length
    u4_difference = [0] * difference_length

    # 1* 25
    def phi_1(k):
        temp = y_output[(k - 1 - 1):(k - na - 1 - 1):-1]
        return np.array([[x * -1 for x in temp] +
                         u1_difference[(k - d1 - 2):(k - d1 - nb1 - 2 - 1):-1] +
                         u2_difference[(k - d2 - 2):(k - d2 - nb2 - 2 - 1):-1] +
                         u3_difference[(k - d3 - 2):(k - d3 - nb3 - 2 - 1):-1] +
                         u4_difference[(k - d4 - 2):(k - d4 - nb4 - 2 - 1):-1]])

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
    error_analysis_plot(y, dfnew['predict_data'],plot_open)
    # print(dfnew.head())

    # print(dfnew.columns)
    # for i in range(start_item, end_item):
    #     pass


if __name__ == '__main__':
    start = clock()
    for delay in range(15, 16):
        print('delay: ', delay)
        analysis1108(delay, True)
    # df1 = pd.read_csv('D:/08_35.csv')  # .dropna()
    # df2 = pd.read_csv('D:/09.csv')  # .dropna()
    #
    # x = df1[['LAECF411', 'u2', 'u3', 'TOTFUELF']]
    # y = df1.T12A041A
    # delay = 10
    # d1 = d2 = d3 = d4 = delay
    #
    # na, nb1, nb2, nb3, nb4 = 5, 10, 0, 0, 0
    # dfnew = pd.DataFrame(columns=['a'])
    # columns = ['_y_k_' + str(k) for k in range(1, na + 1)] + ['deltaU1_k_1_' + str(k) for k in range(0, nb1 + 1)] + [
    #     'deltaU2_k_1_' + str(k) for k in range(0, nb2 + 1)] + ['deltaU3_k_1_' + str(k) for k in range(0, nb3 + 1)] + [
    #               'deltaU4_k_1_' + str(k) for k in range(0, nb4 + 1)]
    # print(columns)
    # def data_create():
    # map(lambda k: dfnew['_y_k_' + str(k)] = y)
    # for k in range(1, na+1):
    #     dfnew['y_'+str(k)] = 1
    # print(dfnew)
    plt.show()

    end = clock()
    print('time: {:.8f}s'.format(end - start))
