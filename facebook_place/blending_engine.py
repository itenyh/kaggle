#coding:utf8

from __future__ import division
import numpy as np
import pandas as pd
from metrics import fb_validate3

all = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
                    usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)


def vlidate_by_model(model):

    df = pd.read_csv(model, index_col = 0)
    df = df.fillna(0)
    cols = df.columns
    ranks_index = np.argsort(df.values, axis=1)[:,::-1][:,:3]
    for r in ranks_index:

        for index, ri in enumerate(r):

            r[index] = cols[ri]

    df['0_'], df['1_'], df['2_'] = zip(*ranks_index)
    df = df[['0_','1_','2_']]
    df['row_id'] = df.index
    df = df.reset_index(drop=True)

    print fb_validate3(df, all)


vlidate_by_model('model/knn-26-base.csv')