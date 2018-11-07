# @Time    : 2018/11/7 11:15
# @Author  : Yanlin Wang
# @Email   : wangyl_a@163.com
# @File    : 偏度.py

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
df['data2'].plot()
plt.show()

end = clock()
print('time: {:.8f}s'.format(end - start))
