# @Time    : 2018/11/7 14:04
# @Author  : Yanlin Wang
# @Email   : wangyl_a@163.com
# @File    : 1. 协方差与相关系数.py

from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start = clock()
# 数据结构, dataframe, 计算每两列的协方差, 相关系数矩阵
df = pd.DataFrame({'data1': np.random.randn(100), 'data2': np.random.randn(100), 'data3': np.random.randn(100)})

# pandas dataframe格式, 返回的是协方差, 方差的矩阵, 一个dataframe格式数据
df1 = df.cov()  # 返回dataframe格式

df2 = df.corr()  # 返回dataframe格式
print(type(df2), df2, sep='\n')

df2.to_csv('a.csv')  # 相关系数矩阵转输出, 索引列也输出

end = clock()
print('time: {:.8f}s'.format(end - start))
