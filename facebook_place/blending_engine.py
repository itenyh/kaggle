#coding:utf8

from __future__ import division
import numpy as np
import pandas as pd
import os, datetime, time, itertools
from metrics import fb_validate4

all = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
                    usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)

def lr(acc):

    err = 1 - acc

    deta_t = np.math.sqrt((1 - err) / err)

    return np.math.log(deta_t)

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

    print 'Now blending : %s' % models

    start_time = time.time()

    indexes = None
    model_list = []
    weight = []

    for m_i, model in enumerate(models):

        m, w = model.split('#')
        df = pd.read_csv('model/' + m, index_col = 0)
        weight.append(float(w))
        if indexes is None: indexes = df.index
        model_list.append(df)

    new_data = []

    len_index = len(indexes)
    i_n = 0
    for row_id in indexes:

        i_n += 1

        if i_n % int(len_index / 100) == 0: print '%d/%d'% (i_n, len_index)

        all_rows = []

        for model in model_list:

            all_rows.append(model.ix[row_id].values)

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

            if not is_bad: new_r.append(list(n))

    return new_r

# m_list = ['p-knn-06251136.csv#0.631837','p-knn-06251138.csv#0.633387','p-knn-06251141.csv#0.634431','p-knn-06251143.csv#0.632588'
# ,'p-knn-06251144.csv#0.635404','p-knn-06251503.csv#0.634376','p-knn-06251447.csv#0.632272','p-knn-06251453.csv#0.633616','p-knn-06251450.csv#0.635072'
# ,'p-knn-06251456.csv#0.63353','p-knn-06251459.csv#0.636322','p-knn-06251504.csv#0.634518','p-knn-06251506.csv#0.648797','p-knn-06251510.csv#0.649216'
# ,'p-knn-06251513.csv#0.649216','p-knn-06251515.csv#0.649501','p-knn-06251517.csv#0.647681','p-knn-06251519.csv#0.647681']

m_list = ['model_1_24_sample/s-knn-06261242.csv#0.6','model_1_24_sample/s-knn-06261357.csv#0.6']

print vlidate_by_model(m_list)

# m_list = blend_combine(m_list, [])
# print(len(m_list))
# scores = []
#
# for i, item in enumerate(m_list):
#
#     print(i + 1)
#     s = vlidate_by_model(item)
#     scores.append((s, item))
#
# scores = sorted(scores, key=lambda x:x[0], reverse=True)
# print(scores)

# print vlidate_by_model(['p-knn-06251144.csv'])

'''全部有效的m_list
m_list = ['p-knn-06251138.csv#0.633387','p-knn-06251143.csv#0.632588'
,'p-knn-06251503.csv#0.634376','p-knn-06251453.csv#0.633616','p-knn-06251456.csv#0.63353','p-knn-06251504.csv#0.634518','p-knn-06251506.csv#0.648797','p-knn-06251510.csv#0.649216'
,'p-knn-06251515.csv#0.649501','p-knn-06251519.csv#0.647681']
'''

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


