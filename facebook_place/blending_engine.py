#coding:utf8

from __future__ import division
import numpy as np
import pandas as pd
import os, datetime, time, itertools
from metrics import fb_validate4


# all = pd.read_table('data/1_4_sample.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
#                     usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)

all = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
                    usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)


def lr(acc):

    err = 1 - acc

    deta_t = np.math.sqrt((1 - err) / err)

    return np.math.log(deta_t)

    return acc

def vlidate_by_model_4_alldata(models, filename = None):

    start_time = time.time()

    indexes = None
    model_list = []
    weight = []

    print 'Loading......'

    for m_i, model in enumerate(models):

        m, w = model.split('#')
        df = pd.read_csv('model/' + m, index_col = 0)
        weight.append(float(w))
        if indexes is None: indexes = df.index
        model_list.append(df.values)

    print 'Merging......'
    new_data = []
    for row_id in indexes:

        all_rows = []

        for model in model_list:

            all_rows.append(model[row_id])

        new_row = merge_rows(all_rows, weight)

        new_data.append(new_row)

        if row_id % 10000 == 0: print row_id

    print 'time eclapse : %.2f second' % (time.time() - start_time)

    print 'Producing ......'

    preds_total = pd.DataFrame(new_data, index=indexes, columns=['0_', '1_', '2_'], dtype=int)
    preds_total.index.names = ['row_id']

    if filename is not None:

        preds_total = preds_total.astype(str)
        ds_sub = preds_total['0_'].str.cat([preds_total['1_'], preds_total['2_']], sep=' ')
        ds_sub.name = 'place_id'
        ds_sub.to_csv(filename, index=True, header=True, index_label='row_id')


def vlidate_by_model(models, filename = None):

    # print 'Now blending : %s' % models

    start_time = time.time()

    indexes = None
    model_list = []
    weight = []

    for m_i, model in enumerate(models):

        m, w = model.split('#')
        df = pd.read_csv('model/' + m, index_col = 0)
        weight.append(float(w))
        if indexes is None: indexes = df.index
        model_list.append(df.values)

    new_data = []

    len_index = len(indexes)
    i_n = 0
    for ii, row_id in enumerate(indexes):

        # i_n += 1
        # if i_n % int(len_index / 100) == 0: print '%d/%d'% (i_n, len_index)

        all_rows = []

        for model in model_list:

            all_rows.append(model[ii])

        new_row = merge_rows(all_rows, weight)

        new_data.append(new_row)

    print 'time eclapse : %.2f second' % (time.time() - start_time)

    preds_total = pd.DataFrame(new_data, index=indexes, columns=['0_', '1_', '2_'], dtype=int)
    preds_total.index.names = ['row_id']

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
    for i in range(top_score):

        for j, item in enumerate(all):

            place_id = str(item[i])

            if score.has_key(place_id): score[place_id] += ((top_score - i) * lr(weight[j]))
            else: score[place_id] = ((top_score - i) * lr(weight[j]))

    score = score.items()
    score = sorted(score, key=lambda x:x[1], reverse=True)[:3]
    score = [s[0] for s in score]

    return score

# ================适合大数据的validate模块===================

def h_merge_dict(row, weight, s_dict):

    top_score = len(row)
    for i in range(top_score):

        place_id = str(row[i])

        if s_dict.has_key(place_id): s_dict[place_id] += (top_score - i) * lr(weight)
        else: s_dict[place_id] = (top_score - i) * lr(weight)

    return s_dict

def h_validate(models, filename = None):

    start_time = time.time()

    indexes = None
    model_list = []
    weight = []

    print 'Loading......'

    all_dict_list = []

    for m_i, model in enumerate(models):

        print 'Now merge model %d ==> %s' % (m_i, model)

        m, w = model.split('#')
        w = float(w)
        df = pd.read_csv('model/' + m, index_col = 0)
        weight.append(float(w))
        if indexes is None: indexes = df.index
        rows = df.values

        for i, row in enumerate(rows):

            if len(all_dict_list) < i + 1 : all_dict_list.append({})
            ss_dict = all_dict_list[i]

            ss_dict = h_merge_dict(row, w, ss_dict)

            all_dict_list[i] = ss_dict

    new_data = []
    print 'Scoring ......'
    for score in all_dict_list:

        score = score.items()
        score = sorted(score, key=lambda x:x[1], reverse=True)[:3]
        score = [s[0] for s in score]

        new_data.append(score)

    print 'time eclapse : %.2f second' % (time.time() - start_time)

    preds_total = pd.DataFrame(new_data, index=indexes, columns=['0_', '1_', '2_'], dtype=int)
    preds_total.index.names = ['row_id']

    s = fb_validate4(preds_total, all)

    if filename is not None:

        preds_total = preds_total.astype(str)
        ds_sub = preds_total['0_'].str.cat([preds_total['1_'], preds_total['2_']], sep=' ')
        ds_sub.name = 'place_id'
        ds_sub.to_csv(filename, index=True, header=True, index_label='row_id')

    return s


# ===================================

def blend_combine(b_list, b_fi = []):

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

            if not is_bad and len(list(n)) > 1 : new_r.append(list(n))

    return new_r

# m_list = ['p-knn-06251136.csv#0.631837','p-knn-06251138.csv#0.633387','p-knn-06251141.csv#0.634431','p-knn-06251143.csv#0.632588'
# ,'p-knn-06251144.csv#0.635404','p-knn-06251503.csv#0.634376','p-knn-06251447.csv#0.632272','p-knn-06251453.csv#0.633616','p-knn-06251450.csv#0.635072'
# ,'p-knn-06251456.csv#0.63353','p-knn-06251459.csv#0.636322','p-knn-06251504.csv#0.634518','p-knn-06251506.csv#0.648797','p-knn-06251510.csv#0.649216'
# ,'p-knn-06251513.csv#0.649216','p-knn-06251515.csv#0.649501','p-knn-06251517.csv#0.647681','p-knn-06251519.csv#0.647681']

m_list = ['model_nine/p-knn-06251456.csv#0.63353', 'model_nine/p-knn-06251510.csv#0.649216', 'model_nine/p-knn-06251504.csv#0.634518',
          'model_nine/p-knn-06251515.csv#0.649501','model_nine/p-knn-06251519.csv#0.647681']
# 0.653780435711
print h_validate(m_list, 'out_test,csv')
exit()

# h-knn-06280809.csv (去掉xyaccuracy) 0.085515
# h-knn-06280823.csv (再加半个圈的时间) 0.084376
# h-knn-06280901.csv# (加上所有,半圈投影) 0.0826977580314
# h-knn-06281131.csv 0.026103 (加上所有,半圈投影,10*10)

# h-knn-06280823.csv#0.084376', 's-knn-1719.csv#0.641612' 0.642632
# h-knn-06280809.csv#0.085515', 's-knn-1719.csv#0.641612  0.641631
# h-knn-06280901.csv#0.082698', 's-knn-1719.csv#0.641612  0.643463
# h-knn-06281131.csv#0.026103', 's-knn-1719.csv#0.641612  0.641611

# h-knn-06280901.csv + h + m + w 0.644418


#时间模型不能bleading,否则影响太大


# print vlidate_by_model(m_list)
# vlidate_by_model_4_alldata(m_list, 'submit/sub-knn-06281530.csv')
'''
m_list = ['s-knn-06271826.csv#0.6402431' ,'s-knn-06271830.csv#0.640175' , 's-knn-06271835.csv#0.637399']
m_list = blend_combine(m_list)
b_N = len(m_list)
print(b_N)

scores = []
print 'Starting ......'
for i, item in enumerate(m_list):

    s = vlidate_by_model(item)
    scores.append((s, item))

    print 'Blending  %d/%d Over, Score: %f' % ((i+1), b_N, s)

scores = sorted(scores, key=lambda x:x[0], reverse=True)
print(scores)


# print vlidate_by_model(['model_1_4_sample/s-knn-06271841.csv#0.5'])


m_list = ['p-knn-06251138.csv#0.633387','p-knn-06251143.csv#0.632588'
,'p-knn-06251503.csv#0.634376','p-knn-06251453.csv#0.633616','p-knn-06251456.csv#0.63353','p-knn-06251504.csv#0.634518','p-knn-06251506.csv#0.648797','p-knn-06251510.csv#0.649216'
,'p-knn-06251515.csv#0.649501','p-knn-06251519.csv#0.647681']
m_list = ['s-knn-1719.csv#0.641612', 's-knn-06271728.csv#0.642232', 's-knn-06271806.csv#0.639685',
's-knn-06271815.csv#0.639756', 's-knn-06271826.csv#0.6402431'
    ,'s-knn-06271830.csv#0.640175' ,'s-knn-06271835.csv#0.637399' ,'s-knn-06271841.csv#0.637762']
'''


