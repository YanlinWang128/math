# @Time    : 2018/11/7 11:15
# @Author  : Yanlin Wang
# @Email   : wangyl_a@163.com
# @File    : 0. 偏度_峰度_基本统计量.py

from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

start = clock()
df = pd.DataFrame({'data1': [3 * x ** 3 + 6 for x in range(-1000, 1000)], 'data2': np.random.randn(2000)})
# print(df)

# 偏度
skew = df['data2'].skew()

# 峰度
kurt = df['data2'].kurt()
print('偏度: {}\n峰度: {}'.format(skew, kurt))

# describe() 对每列series可以进行描述
"""
count    2000.000000    样本量
mean        0.039298    均值
std         0.993351    标准差
min        -2.901006    最小值
25%        -0.655077    25%分位数
50%         0.094904    中位数
75%         0.702151    75%分位数
max         3.506999    最大值
"""

print(df['data2'].describe())

df['data2'].plot()
# plt.show()

end = clock()
print('time: {:.8f}s'.format(end - start))
