#coding:utf8

from __future__ import division
import numpy as np
import pandas as pd
import os, datetime
from metrics import fb_validate3

all = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
                    usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)

def lr(acc):

    err = 1 - acc

    deta_t = np.math.sqrt((1 - err) / err)

    return np.math.log(deta_t)

def vlidate_by_model(models, weight = None, filename = None):

    total_pre = pd.DataFrame()
    for m_i, model in enumerate(models):

        df = pd.read_csv(model, index_col = 0)
        df = df.fillna(0)

        w = 1
        if weight is not None:
            w = lr(weight[m_i])

        if len(total_pre) == 0:
            total_pre = (df * w)
        else:
            total_pre += (df * w)

        # print(total_pre)

    cols = total_pre.columns
    ranks_index = np.argsort(total_pre.values, axis=1)[:,::-1][:,:3]
    for r in ranks_index:

        for index, ri in enumerate(r):

            r[index] = cols[ri]

    total_pre['l1'], total_pre['l2'], total_pre['l3'] = zip(*ranks_index)
    total_pre = total_pre[['l1','l2','l3']]
    total_pre['row_id'] = total_pre.index
    total_pre = total_pre.reset_index(drop=True)

    score = fb_validate3(total_pre, all)

    if filename is not None:

        total_pre.drop('row_id', axis = 1, inplace = True)
        total_pre = total_pre.astype(str)
        ds_sub = total_pre.l1.str.cat([total_pre.l2, total_pre.l3], sep=' ')
        ds_sub.name = 'place_id'
        ds_sub.to_csv(filename, index=True, header=True, index_label='row_id')

    return score



vlidate_by_model(['model/rf-base.csv'], [0.6471], 'test_sub.csv')