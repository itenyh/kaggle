#coding:utf8

from __future__ import division
import numpy as np
import pandas as pd
import os, datetime, time
from metrics import fb_validate4

all = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
                    usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)

def lr(acc):

    err = 1 - acc

    deta_t = np.math.sqrt((1 - err) / err)

    return np.math.log(deta_t)

def vlidate_by_model(models, weight = None, filename = None):

    start_time = time.time()

    indexes = None
    model_list = []
    for m_i, model in enumerate(models):

        df = pd.read_csv('model/' + model, index_col = 0)
        if indexes is None: indexes = df.index
        if weight is None: weight = [1] * len(indexes)
        model_list.append(df)

    new_data = []

    for row_id in indexes:

        all_rows = []

        for model in model_list:

            all_rows.append(model.ix[row_id].values)

        new_row = merge_rows(all_rows, weight)
        new_data.append(new_row)

    preds_total = pd.DataFrame(new_data, index=indexes, columns=['0_', '1_', '2_'], dtype=int)
    preds_total.index.names = ['row_id']

    print 'time eclapse : %.2f second' % (time.time() - start_time)

    score = fb_validate4(preds_total, all)

    if filename is not None:

        preds_total = preds_total.astype(str)
        ds_sub = preds_total['0_'].str.cat([preds_total['1_'], preds_total['2_']], sep=' ')
        ds_sub.name = 'place_id'
        ds_sub.to_csv(filename, index=True, header=True, index_label='row_id')

    return score


def merge_rows(all, weight):

    a = all[0]
    score = {}

    top_score = len(a)
    for i in range(len(a)):

        for j, item in enumerate(all):

            place_id = str(item[i])

            if score.has_key(place_id): score[place_id] += ((top_score - i) * weight[j])
            else: score[place_id] = ((top_score - i) * weight[j])

    score = score.items()
    score = sorted(score, key=lambda x:x[1], reverse=True)[:3]
    score = [s[0] for s in score]

    return score

print vlidate_by_model(['p-knn-split.csv'])

# a = pd.DataFrame(np.random.rand(3,2), columns=['0_', '1_'], index=['a', 'b', 'c'])
# print(a)
# b = pd.DataFrame(np.random.rand(3,2), columns=['0_', '1_'], index=['a', 'b', 'c'])
# print(b)

# print(a + b)

# a = [42, 51, 34]
# b = [51, 42, 21]
# top_score = len(a)
# learn_rate = [1, 0.5]
# all = [a, b]
# score = {}
#
# for i in range(len(a)):
#
#     for j, item in enumerate(all):
#
#         place_id = str(item[i])
#         if score.has_key(place_id): score[place_id] += ((top_score - i) * learn_rate[j])
#         else: score[place_id] = ((top_score - i) * learn_rate[j])
#
# score = score.items()
# score = sorted(score, key=lambda x:x[1], reverse=True)[:3]
# score = [s[0] for s in score]

# a = pd.DataFrame(np.array([[21, 32, 3],[43,53, 0]]), columns=['0_', '1_', '_2'], index=['a', 'b'])
    # print(a)
    # b = pd.DataFrame(np.array([[21, 3, 99],[43,53, 2]]), columns=['0_', '1_', '2_'], index=['a', 'b'])
    # print(b)

    # indexes = a.index
    # model_list = [a, b]


