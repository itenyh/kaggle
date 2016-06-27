#coding:utf8

from __future__ import division
import numpy as np
import pandas as pd
import os, time

data_path = 'data/split_data/'

def test_data(fw, name = 'data/ninegrid_xy.txt'):

    all = pd.read_table(name, sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
                    usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)
    #
    N = len(all)
    df_train = all.iloc[int(0.2 * N):]
    df_test = all.iloc[:int(0.2 * N)]

    df_train, df_test = data_engineering(df_train, df_test, fw)
    #
    # df_train = pd.read_csv('data/train.csv',
    #                        usecols=['row_id','x','y','time','place_id','accuracy'],
    #                        index_col = 0)
    # df_test = pd.read_csv('data/test.csv',
    #                       usecols=['row_id','x','y','time','accuracy'],
    #                       index_col = 0)
    #(8607230, 4)

    return all, df_train, df_test

def data_engineering(df_train, df_test, fw):

    minute = df_train['time']%60
    df_train['hour'] = df_train['time']//60
    df_train['weekday'] = df_train['hour']//24
    df_train['month'] = df_train['weekday']//30
    df_train['year'] = (df_train['weekday']//365+1)*fw[5]
    # df_train['season'] = (((df_train['month'])//3 + 3) % 4 + 1) * fw[7]
    df_train['hour'] = ((df_train['hour']%24+1)+minute/60.0)* fw[2]
    df_train['weekday'] = (df_train['weekday']%7+1)*fw[3]
    df_train['month'] = (df_train['month']%12+1)*fw[4]
    df_train['accuracy'] = np.log10(df_train['accuracy'])*fw[6]
    df_train.drop(['time'], axis=1, inplace=True)
    # print df_train.month.min()
    minute = df_test['time']%60
    df_test['hour'] = df_test['time']//60
    df_test['weekday'] = df_test['hour']//24
    df_test['month'] = df_test['weekday']//30
    df_test['year'] = (df_test['weekday']//365+1)*fw[5]
    # df_test['season'] = (((df_test['month'])//3 + 3) % 4 + 1) * fw[7]
    df_test['hour'] = ((df_test['hour']%24+1)+minute/60.0)*fw[2]
    df_test['weekday'] = (df_test['weekday']%7+1)*fw[3]
    df_test['month'] = (df_test['month']%12+1)*fw[4]
    df_test['accuracy'] = np.log10(df_test['accuracy'])*fw[6]
    df_test.drop(['time'], axis=1, inplace=True)

    #add data for periodic time that hit the boundary
    # add_data = df_train[df_train.hour<6]
    # add_data.hour = add_data.hour + 24 * fw[2]
    # df_train = df_train.append(add_data)
    #
    # add_data = df_train[df_train.hour>98]
    # add_data.hour = add_data.hour - 24 * fw[2]
    # df_train = df_train.append(add_data)

    #add data for periodic week(3) that hit the boundary
    # add_data = df_train[df_train.weekday <= (3 * fw[3])]
    # add_data.weekday = add_data.weekday + 7 * fw[3]
    # df_train = df_train.append(add_data)
    #
    # add_data = df_train[df_train.weekday >= (5 * fw[3])]
    # add_data.weekday = add_data.weekday - 7 * fw[3]
    # df_train = df_train.append(add_data)

    #add data for periodic month(2) that hit the boundary
    # add_data = df_train[df_train.month <= (3 * fw[4])]
    # add_data.month = add_data.month + 12 * fw[4]
    # df_train = df_train.append(add_data)
    #
    # add_data = df_train[df_train.hour >= (10 * fw[4])]
    # add_data.month = add_data.month - 12 * fw[4]
    # df_train = df_train.append(add_data)

    #crazy time config
    # add_data = df_train[df_train.hour <= (5 * fw[2])]
    # add_data.hour = add_data.hour + 24 * fw[2]
    # df_train = df_train.append(add_data)
    #
    # add_data = df_train[df_train.hour >= (19 * fw[2])]
    # add_data.hour = add_data.hour - 24 * fw[2]
    # df_train = df_train.append(add_data)

    return df_train, df_test

def cell_split(df_train, df_test, size, x_step, x_border_augment, sub_fix = ''):

    start_time = time.time()
    all_train_num = 0
    all_test_num = 0

    x_grid_num = (int)(size/x_step)
    dir_name = 'X-' + str(x_grid_num) + '_' + sub_fix + '/'

    if not os.path.exists(data_path + dir_name):
        os.makedirs(data_path + dir_name)
    else:
        print('File exist \n Finish!')
        exit()

    for i in range(x_grid_num):

        row_time = time.time()

        x_min = x_step * i
        x_max = x_step * (i+1)
        x_min = round(x_min, 4)
        x_max = round(x_max, 4)
        if x_max == size:
            x_max = x_max + 0.001

        df_col_train = df_train[(df_train['x'] >= x_min-x_border_augment) & (df_train['x'] < x_max+x_border_augment)]
        df_col_test = df_test[(df_test['x'] >= x_min) & (df_test['x'] < x_max)]

        train_num = len(df_col_train)
        test_num = len(df_col_test)
        all_train_num += train_num
        all_test_num += test_num

        train_file_name = 'train_' + str(i) + '.csv'
        if(len(df_col_train) == 0): train_file_name = 'train_' + str(i) + '_0.csv'
        test_file_name = 'test_' + str(i) + '.csv'
        if(len(df_col_test) == 0): test_file_name = 'test_' + str(i) + '_0.csv'

        df_col_train.to_csv(data_path + dir_name + train_file_name, index=True, header=True, index_label='row_id')
        df_col_test.to_csv(data_path + dir_name + test_file_name, index=True, header=True, index_label='row_id')

        print "Process time: %.2f seconds on row:%d Train:%d, Test:%d" % (time.time() - row_time, i, train_num, test_num)

    print "Total Process time: %.2f seconds Train:%d, Test:%d" % (time.time() - start_time, all_train_num, all_test_num)

def do():

    # #Defining the size of the grid
    size = 10.0
    x_step = 0.5
    y_step = 0.5

    x_border_augment = 0.025
    y_border_augment = 0.025
    #
    # print('Loading data ...')
    # df_train = pd.read_csv('data/train.csv',
    #                        usecols=['row_id','x','y','time','place_id','accuracy'],
    #                        index_col = 0)
    # df_test = pd.read_csv('data/test.csv',
    #                       usecols=['row_id','x','y','time','accuracy'],
    #                       index_col = 0)
    # #
    all = pd.read_table('data/1_4_sample.txt', sep = ',', names = ['row_id', 'x', 'y', 'accuracy', 'time', 'place_id'],
                        usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)

    N = len(all)
    df_train = all.iloc[int(0.2 * N):]
    df_test = all.iloc[:int(0.2 * N)]

    print('Processing ...')
    cell_split(df_train, df_test, size, x_step, x_border_augment, '1_4')


# do()
