#coding:utf8

import numpy as np
import pandas as pd
# from common.tools import *

def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    # print(score, actual, predicted)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def fb_validate1(pred, data_frame):

    """
    :param pred:[[1,2,3],[4,5,6],......] 行数代表row_id
    :return:
    """
    # '''
    #     0_ 1_ 2_ row_id
    # '''
    #

    new_pred = []
    for index, item in enumerate(pred):

        if item[0] == 0 and item[1] == 0 and item[2] == 0:

            continue

        item = np.append(item, [index])
        new_pred.append(item)

    pre_frame = pd.DataFrame(new_pred, columns=['0_', '1_', '2_', 'row_id'])

    pre_frame = pre_frame.sort_values(by=['row_id'])
    pre_list = pre_frame.drop('row_id', axis=1).values

    pre_row_id_list = pre_frame['row_id'].values

    actual_data_frame = data_frame[data_frame.index.isin(pre_row_id_list)]
    acutal_list = actual_data_frame['place_id'].values
    acutal_list = [[v] for v in acutal_list]


    return mapk(acutal_list, pre_list)

def fb_validate2(pre_frame, data_frame):

    pre_frame = pre_frame.sort_values(by=['row_id'])

    print(pre_frame)
    exit()

    pre_list = pre_frame.drop('row_id', axis=1).values

    pre_row_id_list = pre_frame['row_id'].values

    actual_data_frame = data_frame[data_frame["row_id"].isin(pre_row_id_list)]
    acutal_list = actual_data_frame['place_id'].values
    acutal_list = [[v] for v in acutal_list]

    return mapk(acutal_list, pre_list)

def fb_validate(pre_frame, data_frame):

    pre_frame = pre_frame.sort_values(by=['row_id'])
    pre_list = pre_frame.drop('row_id', axis=1).values

    pre_row_id_list = pre_frame['row_id'].values

    actual_data_frame = data_frame[data_frame["row_id"].isin(pre_row_id_list)]
    acutal_list = actual_data_frame['place_id'].values
    acutal_list = [[v] for v in acutal_list]

    return mapk(acutal_list, pre_list)

#在fb_validate基础上，改为了data_frame.index
def fb_validate3(pre_frame, data_frame):

    pre_frame = pre_frame.sort_values(by=['row_id'])
    pre_list = pre_frame.drop('row_id', axis=1).values

    pre_row_id_list = pre_frame['row_id'].values

    actual_data_frame = data_frame[data_frame.index.isin(pre_row_id_list)]
    acutal_list = actual_data_frame['place_id'].values
    acutal_list = [[v] for v in acutal_list]

    return mapk(acutal_list, pre_list)
# all = pd.read_table('data/sample_xy.txt', sep = ',', names = ['row_id', 'x', 'y', 'accuracy', 'time', 'place_id'])
# fb_validate1([[1,2,3],[4,5,6]], all)
# cols = all.columns

# print(all[cols.drop('place_id')])