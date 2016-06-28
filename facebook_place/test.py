#coding:utf8

from __future__ import division
from  common.tools import print_type
import numpy as np
import pandas as pd
import metrics as mt
import os

import sklearn.preprocessing as pp
# import xgboost as xgb
import itertools

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

def max_nonzero_num(t):

    t = np.sort(t, axis = 1)[:,::-1]
    N = len(t)

    total_non_zero_index = 0
    max_num = -1
    for item_i in t:

        non_zero_index = 0

        for item_j in item_i:

            if item_j == 0:

                break

            non_zero_index += 1

            if non_zero_index > max_num:

                max_num = non_zero_index

        total_non_zero_index += non_zero_index

    print max_num

# data = np.array([[0.4, 0.4, 0.8, 0 , 0, 0], [0.2, 0.2, 0.4, 0 , 0, 0]])
# data2 = np.array([[5, 5, 10, 0 , 0, 0], [2, 2, 4, 0 , 0, 0]])
# x = np.array([[0, 3, 1, 0.5], [4, 2, 7, 1]])

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

# d1 = np.array([[1,2], [4,4]])
# d2 = np.array([[3,4], [1, 2]])


# print d2 * 0.1

# dtrain = xgb.DMatrix('data/agaricus.txt.train')
# dtest = xgb.DMatrix('data/agaricus.txt.test')
# # specify parameters via map
# param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
# num_round = 2
# bst = xgb.train(param, dtrain, num_round)
# # make prediction
# preds = bst.predict(dtest)

# pp = pd.DataFrame(np.random.rand(3,2))
# print_type(pp)
# print pp[0] + 1

# row_ids = [5, 21]
# labels = [13, 23]
# values = np.array([[1,2],[3,4]])
#
# pd1 = pd.DataFrame(data = values, index=row_ids, columns=labels)
#
# row_ids = [5, 21]
# labels = [13, 23]
# values = np.array([[9,4],[1,2]])
# pd2 = pd.DataFrame(data = values, index=row_ids, columns=labels)
#
# print(pd.concat([pd1,pd2]))

# t = [(0.3, 0, 9), (0.6, 0, 8)]
# max_nonzero_num(t)
# print sorted(t, key=lambda tuple:tuple[0], reverse=True)

# data = np.random.rand(30, 25000000)
# print('producing')
# ddd = pd.DataFrame(data)
# print(ddd.head())
# print('transpose')
# ddd=ddd.T
# print(ddd.head())

# dic = {'123':[3,6,7], '453':[1,5,3]}
# print pd.DataFrame(dic, index=['0_', '1_', '2_'])


def blend_combine(b_list, b_fi):

    r = []

    for i in range(1,len(b_list)+1):

        iter = itertools.combinations(b_list,i)

        r.append(list(iter))


    new_r = []
    for m in r:

        for n in m:

            is_bad = False

            for item in b_fi:

                in_count = 0

                for itemm in n:

                    if itemm in item:

                        in_count += 1

                if in_count >= 2:

                    is_bad = True
                    break

            if not is_bad: new_r.append(n)

    return new_r


# a = ['a', 'b', 'c', 'd']
#
# fi = [['a', 'b'], ['a', 'c']]
#
# blend_combine(a, fi)

dic = {'a' : [1 , 2 , 3]}
df = pd.DataFrame(dic)

add_data1 = df[df.a < 2]
add_data2 = df[df.a > 2]

add_data1.a += 3
df = df.append(add_data1)
print(add_data1)

add_data2.a -= 3
print(add_data2)
df = df.append(add_data2)

print '========='

print(df)

