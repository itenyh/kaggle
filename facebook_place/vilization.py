#coding:utf8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import six
from common.tools import *

df_train = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y', 'accuracy', 'time', 'place_id'])
print df_train.shape

print('Deal with time ...')
df_train.loc[:, 'hour'] = (df_train['time']//60)%24 + 1
df_train.loc[:, 'weekday'] = (df_train['time']//1440)%7 + 1
df_train.loc[:, 'month'] = (df_train['time']//43200)%12 + 1
df_train.loc[:, 'year'] = (df_train['time']//525600) + 1

current_palette = sns.color_palette()

# part_train = df_train['weekday']
# counts1, binc1 = np.histogram(part_train.values, bins='auto', range=[1, 8])
# bincs1 = binc1[:-1]
#
# plt.bar(bincs1, counts1)
# plt.show()

plt.scatter(df_train['hour'], df_train['accuracy'], s=1, c='k', lw=0, alpha=0.1)
plt.xlim(df_train['hour'].min(), df_train['hour'].max())
plt.ylim(df_train['accuracy'].min(), df_train['accuracy'].max())
plt.show()


