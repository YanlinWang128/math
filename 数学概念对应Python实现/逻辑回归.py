# @Time    : 2018/11/7 20:14
# @Author  : Yanlin Wang
# @Email   : wangyl_a@163.com
# @File    : 逻辑回归.py

from time import clock
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression as LR
from sklearn.linear_model import RandomizedLogisticRegression as RLR
import imp

start = clock()
# 逻辑回归, 自动建模
# 案例, 根据年龄, 教育, 工龄, 收入, 负债率, 信用卡负债率 来预测 是否 违约
# 输入有很多,需要先进行特征筛选, 看看哪些特征 有用, 需要选取

df1 = pd.read_csv('D:/08_35.csv')  # .dropna()
df2 = pd.read_csv('D:/09.csv')  # .dropna()
print(df1.head())
x = df1[['LAECF411', 'u2', 'u3', 'TOTFUELF']]
y = df1.T12A041A
# rlr = RLR().fit(x, y.astype('int')) # 建立随机逻辑回归模型，筛选变量
# print(rlr.get_support() ) # 布尔数组, 获取特征筛选结果，也可以通过.scores_方法获取各个特征的分数
# print(u'通过随机逻辑回归模型筛选特征结束。')
# print(u'有效特征为：%s' % ','.join(df1.columns[rlr.get_support()]))
# x = df1[df1.columns[rlr.get_support()]].as_matrix() #筛选好特征

lr = LR().fit(x, y.astype('int')) #建立逻辑回归模型, sklearn默认的y的类型必须是整数型

print(u'逻辑回归模型训练结束。')
# print(u'模型的平均正确率为：%s' % lr.score(x, y)) #给出模型的平均正确率，本例为81.4%

end = clock()
print('time: {:.8f}s'.format(end - start))
