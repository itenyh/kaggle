#coding: utf-8

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, VotingClassifier
import time
from metrics import fb_validate1

from common.tools import *

fw = [50, 1000, 4, 3, 2, 10]
th = 5

#classification inside one cell
def process_one_cell(df_train, df_test, x_min, x_max, y_min, y_max):

    x_border_augment = 0.025
    y_border_augment = 0.0125

    #Working on df_train
    df_cell_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment) &
                               (df_train['y'] >= y_min-y_border_augment) & (df_train['y'] < y_max+y_border_augment)]
    place_counts = df_cell_train.place_id.value_counts()
    mask = (place_counts[df_cell_train.place_id.values] >= th).values
    df_cell_train = df_cell_train.loc[mask]

    #Working on df_test
    # to be delete: df_cell_test = df_test.loc[df_test.grid_cell == grid_id]
    df_cell_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max) &
                               (df_test['y'] >= y_min) & (df_test['y'] < y_max)]
    row_ids = df_cell_test.index

    if(len(df_cell_train) == 0 or len(df_cell_test) == 0):
        return None, None

    #Feature engineering on x and y
    df_cell_train.loc[:,'x'] *= fw[0]
    df_cell_train.loc[:,'y'] *= fw[1]
    df_cell_test.loc[:,'x'] *= fw[0]
    df_cell_test.loc[:,'y'] *= fw[1]

    #Preparing data
    le = LabelEncoder()
    y = le.fit_transform(df_cell_train.place_id.values)
    X = df_cell_train.drop(['place_id'], axis=1).values.astype(float)

    if 'place_id' in df_cell_test.columns:

        cols = df_cell_test.columns
        cols = cols.drop('place_id')

        X_test = df_cell_test[cols].values.astype(float)

    else:

        X_test = df_cell_test.values.astype(float)

    #Applying the classifier
    # clf = KNeighborsClassifier(n_neighbors=26, weights='distance',
    #                            metric='manhattan')
    clf1 = BaggingClassifier(KNeighborsClassifier(n_neighbors=26, weights='distance',
                                metric='manhattan'), n_jobs=-1, n_estimators=50)
    clf2 = RandomForestClassifier(n_estimators=100, n_jobs=-1)

    eclf = VotingClassifier(estimators=[('lr', clf1), ('rf', clf2)], voting='hard')

    eclf.fit(X, y)
    y_pred = eclf.predict_proba(X_test)
    pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])

    return pred_labels, row_ids

def process_grid(df_train, df_test, size, x_step, y_step, validate = False):

    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """
    N = df_test.index.values.max() + 1

    preds = np.zeros((N, 3), dtype=int)

    for i in range((int)(size/x_step)):
        start_time_row = time.time()
        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4)
        if x_max == size:
            x_max = x_max + 0.001

        for j in range((int)(size/y_step)):

            # print 'Now in the grid x:%d y:%d' % (i, j)

            y_min = y_step * j
            y_max = y_step * (j+1)
            y_min = round(y_min, 4)
            y_max = round(y_max, 4)
            if y_max == size:
                y_max = y_max + 0.001

            #Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell(df_train, df_test, x_min, x_max, y_min, y_max)

            if pred_labels == row_ids == None:
                continue

            #Updating predictions
            preds[row_ids] = pred_labels

            # if validate:
            #     print fb_validate1(preds, df_test)

        print("Row %d/%d elapsed time: %.2f seconds" % (i+1, (int)(size/x_step),(time.time() - start_time_row)))

    print fb_validate1(preds, df_test)
    print 'Finish!'

"""
    print('Generating submission file ...')
    #Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])

    #Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')

    #Writting to csv
    ds_sub.name = 'place_id'
    ds_sub.to_csv('sub_knn.csv', index=True, header=True, index_label='row_id')
"""

pd.options.mode.chained_assignment = None  # default='warn'

print('Loading data ...')
# df_train = pd.read_csv('data/train.csv',
#                            usecols=['row_id','x','y','time','place_id'],
#                            index_col = 0)
# df_test = pd.read_csv('data/test.csv',
#                           usecols=['row_id','x','y','time'],
#                           index_col = 0)

all = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
                    usecols=['row_id', 'x', 'y', 'time', 'place_id'], index_col = 0)
# all = all.sort_values(by='time')

N = len(all)
df_train = all.iloc[int(0.2 * N):]
df_test = all.iloc[:int(0.2 * N)]
validate = True

#Defini# ng the size of the grid
size = 10.0
x_step = 0.5
y_step = 0.5

print('Preparing train data...')
#Feature engineering
mintue = df_train['time']%60
df_train['hour'] = df_train['time']//60
df_train['weekday'] = df_train['hour']//24
df_train['month'] = df_train['weekday']//30
df_train['year'] = (df_train['weekday']//365+1)*fw[5]
df_train['hour'] = ((df_train['hour']%24+1)+mintue/60.0)*fw[2]
df_train['weekday'] = (df_train['weekday']%7+1)*fw[3]
df_train['month'] = (df_train['month']%12+1)*fw[4]
df_train.drop(['time'], axis=1, inplace=True)

print('Preparing test data...')
mintue = df_test['time']%60
df_test['hour'] = df_test['time']//60
df_test['weekday'] = df_test['hour']//24
df_test['month'] = df_test['weekday']//30
df_test['year'] = (df_test['weekday']//365+1)*fw[5]
df_test['hour'] = ((df_test['hour']%24+1)+mintue/60.0)*fw[2]
df_test['weekday'] = (df_test['weekday']%7+1)*fw[3]
df_test['month'] = (df_test['month']%12+1)*fw[4]
df_test.drop(['time'], axis=1, inplace=True)

#Solving classification problems inside each grid cell
print('Training start')
process_grid(df_train, df_test, size, x_step, y_step, validate)
