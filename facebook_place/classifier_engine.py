#coding:utf8

from __future__ import division
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

import time
from metrics import fb_validate3

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

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

    # print(pred_frame)
    # 1327075245  1447458772  1590689183  1785603962  1912601713  \
# row_id
# 18786      0.000000    0.000000    0.000000    0.000000    0.000000
# 37327      0.000000    0.000000    0.000000    0.000000    0.103789

    return pred_frame

def process_grid(df_train, df_test, size, x_step, y_step, x_border_augment, y_border_augment, fw, th):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """

    run_time = 1

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

            if run_time > 1:

                break

            # run_time += 1

            r_state = 0
            #Applying classifier to one grid cell
            clf_knn = KNeighborsClassifier(n_neighbors=36, weights='distance',
                               metric='manhattan', n_jobs = -1)
            #
            # clf_rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=r_state)

            cell_df = process_one_cell(clf_knn, df_cell_train, df_cell_test, fw, th)
            preds_total = pd.concat([preds_total, cell_df], axis=0)

            print "x,y %d,%d elapsed time: %.2f seconds" % (i, j, time.time() - start_time_cell)

        print("Row %d/%d elapsed time: %.2f seconds" % (i+1, (int)(size/x_step),(time.time() - start_time_row)))

    print("process time: %.2f seconds" % (time.time() - process_time))

    # preds_total = preds_total.fillna(0)

    preds_total.to_csv('model/knn-36-base.csv', index=True, header=True, index_label='row_id')

    exit()

    cols = preds_total.columns
    ranks_index = np.argsort(preds_total.values, axis=1)[:,::-1][:,:3]
    for r in ranks_index:

        for index, ri in enumerate(r):

            r[index] = cols[ri]

    preds_total['0_'], preds_total['1_'], preds_total['2_'] = zip(*ranks_index)
    preds_total = preds_total[['0_','1_','2_']]
    preds_total['row_id'] = preds_total.index
    preds_total = preds_total.reset_index(drop=True)

    print fb_validate3(preds_total, df_test)
    print 'Finish!'

##########################################################
# Main
if __name__ == '__main__':

    # Input varialbles
    fw = [500., 1000., 4., 3., 2., 10., 10.] #feature weights (black magic here)
    th = 5 #Keeping place_ids with more than th samples.

    #Defining the size of the grid
    size = 10.0
    x_step = 0.5
    y_step = 0.5

    x_border_augment = 0.025
    y_border_augment = 0.025

    print('Loading data ...')
    # df_train = pd.read_csv('data/train.csv',
    #                        usecols=['row_id','x','y','time','place_id','accuracy'],
    #                        index_col = 0)
    # df_test = pd.read_csv('data/test.csv',
    #                       usecols=['row_id','x','y','time','accuracy'],
    #                       index_col = 0)

    all = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
                    usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)

    N = len(all)
    df_train = all.iloc[int(0.2 * N):]
    df_test = all.iloc[:int(0.2 * N)]
    validate = True

    #Feature engineering
    print('Preparing train data')
    minute = df_train['time']%60
    df_train['hour'] = df_train['time']//60
    df_train['weekday'] = df_train['hour']//24
    df_train['month'] = df_train['weekday']//30
    df_train['year'] = (df_train['weekday']//365+1)*fw[5]

    df_train['hour'] = ((df_train['hour']%24+1)+minute/60.0)
    df_train['hour'] = df_train['hour'] * fw[2]

    df_train['weekday'] = (df_train['weekday']%7+1)*fw[3]
    df_train['month'] = (df_train['month']%12+1)*fw[4]
    df_train['accuracy'] = np.log10(df_train['accuracy'])*fw[6]
    df_train.drop(['time'], axis=1, inplace=True)


    print('Preparing test data')
    minute = df_test['time']%60
    df_test['hour'] = df_test['time']//60
    df_test['weekday'] = df_test['hour']//24
    df_test['month'] = df_test['weekday']//30
    df_test['year'] = (df_test['weekday']//365+1)*fw[5]
    df_test['hour'] = ((df_test['hour']%24+1)+minute/60.0)*fw[2]
    df_test['weekday'] = (df_test['weekday']%7+1)*fw[3]
    df_test['month'] = (df_test['month']%12+1)*fw[4]
    df_test['accuracy'] = np.log10(df_test['accuracy'])*fw[6]
    df_test.drop(['time'], axis=1, inplace=True)

    print('Solving')
    process_grid(df_train, df_test, size, x_step, y_step, x_border_augment, y_border_augment, fw, th)