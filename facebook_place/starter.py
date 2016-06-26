#coding:utf-8

from __future__ import division
import pandas as pd
import numpy as np
from common.tools import *
import time, datetime
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from metrics import fb_validate

def create_hdf5():

    store = pd.HDFStore('train.h5', mode='w')
    data_reader = pd.read_csv('./train.csv', chunksize=1e8)

    for i, chuck in enumerate(data_reader):

        store.append('all_train', chuck)
        print(i)

    store.close()

def sample_data(input_path, out_path, cutoff = 0.1):

    with open(input_path) as f1:
        with open(out_path, 'w') as f2:
            for i, line in enumerate(f1):
                if np.random.randint(0, 10000) / 10000 < cutoff:
                    f2.write(line)

def sample_data_xy(input_path, out_path, boundlist):

    """

    :param input_path:
    :param out_path:
    :param boundlist: [[(x_min,x_max), (y_min, y_max)], [], []]
    :return:
    """

    count = 0

    with open(input_path) as f1:
        with open(out_path, 'w') as f2:
            for i, line in enumerate(f1):

                if i == 0: continue

                item = line.split(',')
                x = float(item[1])
                y = float(item[2])

                is_in_bound = False

                for bound in boundlist:

                    x_min = bound[0][0]
                    x_max = bound[0][1]
                    y_min = bound[1][0]
                    y_max = bound[1][1]

                    if(x > x_min and x < x_max and y > y_min and y < y_max):

                        is_in_bound = True
                        break


                if is_in_bound:

                    f2.write(line)
                    count += 1

                if i % 1000000 == 0:

                    print 'Now sampling At %d' % i

    return count

def sample_data_uniformity():

    step = 0.2


    pass

def sample_data_ninegrid(input_path, out_path):

    step = 0.2
    mid_per = 1.6

    x_mid = [mid_per, round(mid_per * 3,1), mid_per * 5]
    y_mid = [mid_per, round(mid_per * 3,1), mid_per * 5]

    bound_list = []
    for x_max in x_mid:

        x_min = x_max - step

        for y_max in y_mid:

            y_min = y_max - step

            x_tuple = (round(x_min, 1), x_max)
            y_tuple = (round(y_min, 1), y_max)

            # print(x_tuple)

            bound_list.append([x_tuple, y_tuple])

    # print(bound_list)
    # exit()

    sample_data_xy(input_path, out_path, bound_list)

# sample_data_ninegrid('data/train.csv', 'data/ninegrid_xy.txt')
# print sample_data_xy('data/train.csv', 'data/s_xy.txt', [[(0.0, 0.2), (0.0, 0.2)]])

sample_data('data/train.csv', 'data/1_24_sample.txt', 0.04167)
exit()

print 'Loading data ...'
# all = pd.read_csv('data/train.csv', usecols=['row_id','x','y','accuracy','time','place_id'])
all = pd.read_table('data/sample_xy.txt', sep = ',', names = ['row_id', 'x', 'y', 'accuracy', 'time', 'place_id'])
all = all.sort_values(by=['time'])
# print all
# exit()

N = len(all)
train = all.iloc[:int(0.95 * N)]
test = all.iloc[int(0.95 * N):]

# train = all
# test = all

start_time = time.time()

size = 10.0

x_step = 0.2
y_step = 0.2

empty_grid = 0

x_ranges = zip(np.arange(0, size, x_step), np.arange(x_step, size + x_step, x_step))
y_ranges = zip(np.arange(0, size, y_step), np.arange(y_step, size + y_step, y_step))

print('Deal with time ...')
train.loc[:, 'hour'] = (train['time']//60)%24 + 1
train.loc[:, 'weekday'] = (train['time']//1440)%7 + 1
train.loc[:, 'month'] = (train['time']//43200)%12 + 1
train.loc[:, 'year'] = (train['time']//525600) + 1

test.loc[:, 'hour'] = (test['time']//60)%24 + 1
test.loc[:, 'weekday'] = (test['time']//1440)%7 + 1
test.loc[:, 'month'] = (test['time']//43200)%12 + 1
test.loc[:, 'year'] = (test['time']//525600) + 1

print 'Shape after time engineering'
print train.shape
print test.shape
# exit()
# X_train = train[['x','y','accuracy','time', 'hour', 'weekday', 'month', 'year']]
# y_train = train[['place_id']]
# X_test = test[['x','y','accuracy','time', 'hour', 'weekday', 'month', 'year']]
# X_test_labels = test[['row_id']]

preds_total = pd.DataFrame()
for x_min, x_max in x_ranges:
    start_time_row = time.time()
    for y_min, y_max in y_ranges:

        start_time_cell = time.time()

        x_max = round(x_max, 4)
        x_min = round(x_min, 4)

        y_max = round(y_max, 4)
        y_min = round(y_min, 4)

        if x_max == size:
            x_max = x_max + 0.001

        if y_max == size:
            y_max = y_max + 0.001

        train_grid = train[(train['x'] >= x_min) &
                           (train['x'] < x_max) &
                           (train['y'] >= y_min) &
                           (train['y'] < y_max)]


        test_grid = test[(test['x'] >= x_min) &
                         (test['x'] < x_max) &
                         (test['y'] >= y_min) &
                         (test['y'] < y_max)]

        print x_min, x_max, y_min, y_max
        print train_grid.shape, test_grid.shape

        X_train_grid = train_grid[['x','y','accuracy','time', 'hour', 'weekday', 'month', 'year']]
        y_train_grid = train_grid[['place_id']].values.ravel()
        X_test_grid = test_grid[['x','y','accuracy','time', 'hour', 'weekday', 'month', 'year']]

        if(X_test_grid.shape[0] == 0 or X_train_grid.shape[0] == 0):

            empty_grid += 1
            continue


        # clf = LogisticRegression(n_jobs=-1)
        # clf = SVC(probability=True)
        clf = KNeighborsClassifier(n_neighbors=50, weights='distance',
                               metric='manhattan')
        # clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)

        train_time = time.time()
        clf.fit(X_train_grid, y_train_grid)
        print 'train time %f' % (time.time() - train_time)

        '''
        # clf.predict_proba(X_test_grid) 每行表示1个item属于某个pid的概率（共有class这么多列），共有N行
        # zip(*clf.predict_proba(X_test_grid))
        # [el for el in clf.classes_] pid的list
        # 整个表示class_id：对应的概率列表
        '''


        predict_time = time.time()
        preds = dict(zip([el for el in clf.classes_], zip(*clf.predict_proba(X_test_grid))))
        print 'predict time %f'% (time.time() - predict_time)

        # exit()

        preds = pd.DataFrame.from_dict(preds)

        print(preds)
        exit()

        preds['0_'], preds['1_'], preds['2_'] = zip(*preds.apply(lambda x: preds.columns[x.argsort()[::-1][:3]].tolist(), axis=1))
        preds = preds[['0_','1_','2_']]

        preds['row_id'] = test_grid['row_id'].reset_index(drop=True)

        print "Validate In row: %f" % fb_validate(preds, test)
        preds_total = pd.concat([preds_total, preds], axis=0)

        print "Elapsed time cell: %s seconds == x:%f ~ %f y:%f ~ %f" % (time.time() - start_time_cell, x_min, x_max, y_min, y_max)


    print("Elapsed time row: %s seconds" % (time.time() - start_time_row))

    if(preds_total.shape[0] != 0):

        print "Validate : %f" % fb_validate(preds_total, test)



    # print preds_total

# print X_test_labels.shape
# print preds_total.shape
# sub_file = os.path.join('submission$$_' + str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")) + '.csv')
# preds_total.to_csv(sub_file)

# print fb_validate(preds_total, test)
# print "Elapsed time overall: %s seconds, Empty grid count: %d " % ((time.time() - start_time), empty_grid)

# Elapsed time row: 187.76784682273865 seconds
# Elapsed time row: 218.19643712043762 seconds
# Elapsed time row: 216.88734102249146 seconds
# Elapsed time row: 218.5039827823639 seconds
# Elapsed time row: 211.2776119709015 seconds

# RF:1000
# train time %f 70.7888560295
# predict time %f 39.3959879875

# KNN:50
# train time 0.015942
# predict time 0.155225

# SVC:kbf
# train time 273.933511
# predict time 103.339844
