#coding:utf8

from __future__ import division
import numpy as np
import pandas as pd
from metrics import fb_validate3

all = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
                    usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)


def vlidate_by_model(models):

    total_pre = pd.DataFrame()
    for model in models:

        df = pd.read_csv(model, index_col = 0)
        df = df.fillna(0)

        print(df.head())
        total_pre += df


    cols = total_pre.columns
    ranks_index = np.argsort(total_pre.values, axis=1)[:,::-1][:,:3]
    for r in ranks_index:

        for index, ri in enumerate(r):

            r[index] = cols[ri]

    total_pre['0_'], total_pre['1_'], total_pre['2_'] = zip(*ranks_index)
    total_pre = df[['0_','1_','2_']]
    total_pre['row_id'] = total_pre.index
    total_pre = total_pre.reset_index(drop=True)

    print fb_validate3(total_pre, all)



vlidate_by_model(['model/knn-26-base.csv', 'model/knn-36-base.csv'])