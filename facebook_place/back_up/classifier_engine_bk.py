#coding:utf8

from __future__ import division
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier

from facebook_place.data_engine import data_engineering, test_data
import time, os

from facebook_place.test import max_nonzero_num
from facebook_place.metrics import fb_validate3

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

r_state = 0

def score(total_pre):

    total_pre = total_pre.fillna(0)

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

    return score

# Classification inside one grid cell.
def process_one_cell(clf, df_cell_train, df_cell_test, fw, th):

    #Working on df_train
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]

    #Working on df_test
    row_ids = df_cell_test.index

    #Feature engineering on x and y
    df_cell_train.loc[:,'x'] *= fw[0]
    df_cell_train.loc[:,'y'] *= fw[1]
    df_cell_test.loc[:,'x'] *= fw[0]
    df_cell_test.loc[:,'y'] *= fw[1]

    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id'], axis=1).values

    if 'place_id' in df_cell_test.columns:

        cols = df_cell_test.columns
        cols = cols.drop('place_id')

        X_test = df_cell_test[cols].values.astype(float)

    else:

        X_test = df_cell_test.values.astype(float)


    clf.fit(X, y)
    y_pred = clf.predict_proba(X_test)

    labels = le.inverse_transform(np.arange(y.min(), y.max() + 1))

    pred_frame = pd.DataFrame(data = y_pred, index=row_ids, columns=labels)

    return pred_frame

def process_all(clf, df_train, df_test, size, x_step, y_step, x_border_augment, y_border_augment, fw, th,
                 model_name = 'auto_name.csv', output_model = False):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """

    process_time = time.time()

    preds_total = pd.DataFrame()

    for i in range((int)(size/x_step)):
        start_time_row = time.time()
        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4)
        if x_max == size:
            x_max = x_max + 0.001

        df_col_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment)]
        df_col_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max)]

        for j in range((int)(size/y_step)):

            start_time_cell = time.time()
            y_min = y_step * j
            y_max = y_step * (j+1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)
            if y_max == size:
                y_max = y_max + 0.001

            df_cell_train = df_col_train[(df_col_train['y'] >= y_min-y_border_augment) & (df_col_train['y'] < y_max+y_border_augment)]
            df_cell_test = df_col_test[(df_col_test['y'] >= y_min) & (df_col_test['y'] < y_max)]

            if(len(df_cell_train) == 0 or len(df_cell_test) == 0):
                continue

            cell_df = process_one_cell(clf, df_cell_train, df_cell_test, fw, th)

            add_pre_time = time.time()
            preds_total = pd.concat([preds_total, cell_df], axis=0)


            print "x,y %d,%d elapsed time: %.2f seconds add_pre_time: %.2f train:%d test:%d" % \
                  (i, j, time.time() - start_time_cell, time.time() - add_pre_time, len(df_cell_train), len(df_cell_test))

        print("Row %d/%d elapsed time: %.2f seconds" % (i+1, (int)(size/x_step),(time.time() - start_time_row)))

    print("process time: %.2f seconds" % (time.time() - process_time))

    if output_model: preds_total.to_csv(model_name, index=True, header=True, index_label='row_id')

    return score(preds_total)

def process_split(clf, split_data_file, y_step, y_border_augment, fw, th,
                  model_name = 'auto_name.csv', output_model = False):

    dir_path = 'data/split_data/' + split_data_file + '/'

    process_time = time.time()
    preds_total = pd.DataFrame()
    for i in range(9999):

        start_time_row = time.time()

        train_file_name = 'train_' + str(i) + '.csv'
        if not os.path.exists(dir_path + train_file_name):
            train_file_name = 'train_' + str(i) + '_0.csv'
            if not os.path.exists(dir_path + train_file_name):

                print('Complete!')
                break

        test_file_name = 'test_' + str(i) + '.csv'
        if not os.path.exists(dir_path + test_file_name):
            test_file_name = 'test_' + str(i) + '_0.csv'
            if not os.path.exists(dir_path + test_file_name):
                print('Warning, should not complete with unknow test file name : $s', test_file_name)
                break

        df_col_train = pd.read_csv(dir_path + train_file_name,
                               usecols=['row_id','x','y','time','place_id','accuracy'],
                               index_col = 0)
        df_col_test = pd.read_csv(dir_path + test_file_name,
                              usecols=['row_id','x','y','time','accuracy'],
                              index_col = 0)

        if(len(df_col_train) == 0 or len(df_col_test) == 0):
                continue

        for j in range((int)(size/y_step)):

                start_time_cell = time.time()

                y_min = y_step * j
                y_max = y_step * (j+1)
                y_min = round(y_min, 4)
                y_max = round(y_max, 4)
                if y_max == size:
                    y_max = y_max + 0.001

                df_cell_train = df_col_train[(df_col_train['y'] >= y_min-y_border_augment) & (df_col_train['y'] < y_max+y_border_augment)]
                df_cell_test = df_col_test[(df_col_test['y'] >= y_min) & (df_col_test['y'] < y_max)]

                if(len(df_cell_train) == 0 or len(df_cell_test) == 0):
                    continue

                df_cell_train, df_cell_test = data_engineering(df_cell_train, df_cell_test, fw)

                cell_df = process_one_cell(clf, df_cell_train, df_cell_test, fw, th)
                preds_total = pd.concat([preds_total, cell_df], axis=0)

                print "x,y %d,%d elapsed time: %.2f seconds" % (i, j, time.time() - start_time_cell)

        print("Row %d/%d elapsed time: %.2f seconds" % (i+1, (int)(size/x_step),(time.time() - start_time_row)))

    print("process time: %.2f seconds" % (time.time() - process_time))

    if output_model: preds_total.to_csv(model_name, index=True, header=True, index_label='row_id')

    print score(preds_total)



##########################################################
# Main
if __name__ == '__main__':

    # Input varialbles
    fw = [500., 1000., 4., 3., 2., 10., 10., 4.] #feature weights (black magic here)
    # fw = [1., 1., 1., 1., 1., 1., 1.]
    th = 5 #Keeping place_ids with more than th samples.

    #Defining the size of the grid
    size = 10.0
    x_step = 0.5
    y_step = 0.5

    x_border_augment = 0.025
    y_border_augment = 0.025


    clf_knn = KNeighborsClassifier(n_neighbors=26, weights='distance',
                               metric='manhattan', n_jobs = -1)
    clf_bagging_knn = BaggingClassifier(clf_knn, n_jobs=-1, n_estimators=50, random_state=r_state)
    clf_rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=r_state)

    print('Solving')

    all, df_train, df_test = test_data(fw)
    sc = process_all(clf_rf, df_train, df_test, size, x_step, y_step, x_border_augment, y_border_augment, fw, th
                     , model_name = 'model/p-knn-1658.csv', output_model = False)
    print(sc)


    # result = []
    # ii = range(1, 21)
    # for i in ii:
    #
    #     print '================= %d =================' % i
    #
    #     fw_plus = fw + [float(i)]
    #
    #     all, df_train, df_test = test_data(fw_plus)
    #     sc = process_all(clf_knn, df_train, df_test, size, x_step, y_step, x_border_augment, y_border_augment, fw_plus, th
    #                  , model_name = 'model/p-rf-1435.csv', output_model = False)
    #     result.append(sc)
    #
    # print sorted(zip(result, ii), key=lambda tuple:tuple[0], reverse=True)

    # process_split(clf_knn, 'X-20_all', y_step, y_border_augment, fw, th, model_name = 'model/rf-all.csv', output_model = False)