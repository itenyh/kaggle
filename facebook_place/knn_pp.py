
from __future__ import division
import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
# import xgboost as xgb

from metrics import fb_validate1

import warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

def lr(acc):

    err = 1 - acc

    deta_t = np.math.sqrt((1 - err) / err)

    return np.math.log(deta_t)


# Classification inside one grid cell.
def process_one_cell(df_cell_train, df_cell_test, fw, th):

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

    #Applying the classifier

    r_state = 0

    clf_knn = KNeighborsClassifier(n_neighbors=26, weights='distance',
                               metric='manhattan', n_jobs = -1)
    clf_bagging_knn = BaggingClassifier(clf_knn, n_jobs=-1, n_estimators=50, random_state=r_state)

    clf_rf = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=r_state)

    # clf_gbc = GradientBoostingClassifier(n_estimators=10,  random_state=r_state)

    # clf_nn = Classifier(layers=[Layer('Tanh', units=50), Layer("Softmax")], learning_rate=0.01, n_iter = 125, random_state=r_state)
    # num_round = 2
    # param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
    # bst = xgb.train(param, ,label = y)

    clf_list = [clf_bagging_knn, clf_rf]
    weight = [lr(0.64), lr(0.65)]

    y_pred_all = []

    for ci, cl in enumerate(clf_list):

        cl.fit(X, y)
        y_pred = cl.predict_proba(X_test)

        sort_index = np.argsort(y_pred, axis=1)[:,::-1]

        for j in range(0, len(y_pred)):
            for i, si in enumerate(sort_index[j]):
                y_pred[j][si] = i

        if len(y_pred_all) == 0:

            y_pred_all = np.zeros(y_pred.shape)

        y_pred_all += (y_pred * weight[ci])

    # pred_labels = le.inverse_transform(np.argsort(y_pred, axis=1)[:,::-1][:,:3])
    pred_labels = le.inverse_transform(np.argsort(y_pred_all, axis=1)[:,:3])



    return pred_labels, row_ids

def process_grid(df_train, df_test, size, x_step, y_step, x_border_augment, y_border_augment, fw, th):
    """
    Iterates over all grid cells, aggregates the results and makes the
    submission.
    """
    # preds = np.zeros((df_test.shape[0], 3), dtype=int)

    process_time = time.time()

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

            # print i, j

            if(len(df_cell_train) == 0 or len(df_cell_test) == 0):
                continue

            #Applying classifier to one grid cell
            pred_labels, row_ids = process_one_cell(df_cell_train, df_cell_test, fw, th)

            #Updating predictions
            preds[row_ids] = pred_labels

            print "x,y %d,%d elapsed time: %.2f seconds" % (i, j, time.time() - start_time_cell)

        print("Row %d/%d elapsed time: %.2f seconds" % (i+1, (int)(size/x_step),(time.time() - start_time_row)))

    print("process time: %.2f seconds" % (time.time() - process_time))

    # print fb_validate1(preds, df_test)
    print 'Finish!'


    print('Generating submission file ...')
    #Auxiliary dataframe with the 3 best predictions for each sample
    df_aux = pd.DataFrame(preds, dtype=str, columns=['l1', 'l2', 'l3'])

    #Concatenating the 3 predictions for each sample
    ds_sub = df_aux.l1.str.cat([df_aux.l2, df_aux.l3], sep=' ')

    #Writting to csv
    ds_sub.name = 'place_id'
    ds_sub.to_csv('sub_bagknn-rf-season_pp.csv', index=True, header=True, index_label='row_id')


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
    df_train = pd.read_csv('data/train.csv',
                           usecols=['row_id','x','y','time','place_id','accuracy'],
                           index_col = 0)
    df_test = pd.read_csv('data/test.csv',
                          usecols=['row_id','x','y','time','accuracy'],
                          index_col = 0)
    #
    # all = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
    #                 usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)
    #
    # N = len(all)
    # df_train = all.iloc[int(0.2 * N):]
    # df_test = all.iloc[:int(0.2 * N)]
    # validate = True
                          
    #Feature engineering
    
    print('Preparing train data')
    minute = df_train['time']%60
    df_train['hour'] = df_train['time']//60
    df_train['weekday'] = df_train['hour']//24
    df_train['month'] = df_train['weekday']//30
    df_train['season'] = (df_train['month'] + 2)//3 % 4
    df_train['year'] = (df_train['weekday']//365+1)*fw[5]

    df_train['hour'] = ((df_train['hour']%24+1)+minute/60.0)
    # add_hour_data = df_train[df_train.hour < (6 / fw[2])]
    # add_hour_data.hour = add_hour_data + 24
    # df_train = df_train.append(add_hour_data)
    #
    # add_hour_data = df_train[df_train.hour > (98 / fw[2])]
    # add_hour_data.hour = add_hour_data.hour - 24
    # df_train = df_train.append(add_hour_data)
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
    df_test['season'] = (df_test['month'] + 2)//3 % 4
    df_test['year'] = (df_test['weekday']//365+1)*fw[5]
    df_test['hour'] = ((df_test['hour']%24+1)+minute/60.0)*fw[2]
    df_test['weekday'] = (df_test['weekday']%7+1)*fw[3]
    df_test['month'] = (df_test['month']%12+1)*fw[4]
    df_test['accuracy'] = np.log10(df_test['accuracy'])*fw[6]
    df_test.drop(['time'], axis=1, inplace=True)

    #season: +0 => 0.6319 +1 => 0.632 +2 => 0.6338 +3 => 0.6325

    # print df_train.describe()
    # exit()

    # add data for periodic time that hit the boundary
    # add_data = df_train[df_train.hour<6]
    # add_data.hour = add_data.hour + 24 * fw[2]
    # df_train = df_train.append(add_data)
    #
    # add_data = df_train[df_train.hour>98]
    # add_data.hour = add_data.hour - 24 * fw[2]
    # df_train = df_train.append(add_data)

    print('Solving')
    #Solving classification problems inside each grid cell 
    process_grid(df_train, df_test, size, x_step, y_step, x_border_augment, y_border_augment, fw, th)