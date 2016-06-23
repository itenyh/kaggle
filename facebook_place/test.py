#coding:utf8

from __future__ import division
from  common.tools import print_type
import numpy as np
import pandas as pd
import metrics as mt

import sklearn.preprocessing as pp
import xgboost as xgb

data = np.array([[0.4, 0.4, 0.8, 0 , 0, 0], [0.2, 0.2, 0.4, 0 , 0, 0]])
data2 = np.array([[5, 5, 10, 0 , 0, 0], [2, 2, 4, 0 , 0, 0]])
x = np.array([[0, 3, 1, 0.5], [4, 2, 7, 1]])

# ppp = pp.MaxAbsScaler()
# data_f = ppp.fit_transform(zip(data))

# print(x[0, 3])
# print_type(np.argsort(x, axis=1))

# data2_f = ppp.transform(data2)

# t = np.array([[17, 12 , 31, 6, 0, 0], [12, 17 , 31, 6, 0, 0]])
# sort_index = np.argsort(t, axis=1)
#
# for j in range(0, len(t)):
#     for i, si in enumerate(sort_index[j]):
#
#         t[j][si] = i
#
# print_type(t)

d1 = np.array([[1,2], [4,4]])
d2 = np.array([[3,4], [1, 2]])


print d2 * 0.1

# dtrain = xgb.DMatrix('data/agaricus.txt.train')
# dtest = xgb.DMatrix('data/agaricus.txt.test')
# # specify parameters via map
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)

pp = pd.DataFrame(np.random.rand(3,2))
print_type(pp)
print pp[0] + 1

