#coding:utf8

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression

lr = LogisticRegression()

recent_train = pd.read_table('data/ninegrid_xy.txt', sep = ',', names = ['row_id', 'x', 'y',  'accuracy', 'time', 'place_id'],
                    usecols=['row_id', 'x', 'y', 'accuracy','time', 'place_id'], index_col = 0)

#select a single x_y_grid at random
recent_train = recent_train[(recent_train["x"]>0) &(recent_train["x"]<2) &(recent_train["y"]>0) &(recent_train["y"]<2)]


#derive some features
# recent_train["x"],recent_train["y"] = recent_train["x"]*1000,recent_train["y"]*1000
recent_train["hour"] = recent_train["time"]//60
recent_train["hour_of_day"] = recent_train["hour"]%24 + 1

recent_train["day"] = recent_train["hour"]//24
recent_train["day_of_week"] = recent_train["day"]%7 + 1

recent_train["month"] = recent_train["day"]//30 + 1
recent_train["month_of_year"] = (recent_train["month"]-1)%12 + 1

recent_train["sine"] = np.sin(2*np.pi*recent_train["hour_of_day"]/24)
recent_train["cos"] = np.cos(2*np.pi*recent_train["hour_of_day"]/24)

recent_train["year"] = recent_train["day"]//365 + 1

print("recent_train created")

test = recent_train.sample(axis = 0, frac = 0.05)

print('selected_part and  test created')

features = ["x","y","hour_of_day","day_of_week","month_of_year","year","sine","cos","accuracy"]
constant = [0,0,0,0,0,0,0,0,0]

print (len(test))

colname = str(features)
test[colname] = list
index = test.index
test['done'] = 0

for i in index:

    new_ld = abs(recent_train[features] - test.loc[i][features])
    new_ld.drop(i, inplace=True)
    new_ld["target"] = (recent_train["place_id"] != test.loc[i]["place_id"]) + 0

    #select 100 nearest points based on x+2y distance
    new_ld["x+y"] = (new_ld["x"])+(2*new_ld["y"])
    new_ld = new_ld.sort_values(by="x+y")[0:100]
    true = new_ld[new_ld["target"] == 0]
    false = new_ld[new_ld["target"] != 0]
    #check for skewness
    if (len(true)< 20) | (len(false)< 20):
        # print ("skipped test sample -",i)
        continue

    #get the multipliers which can distinguish between 0 and 1
    lr.fit(new_ld[features], new_ld["target"])
    test.set_value(i, colname, np.maximum(constant, lr.coef_.ravel()))

    test.set_value(i,"done",1)
    print ("done test sample",i)

#average or sum all the multipliers to get overall multiplier
actual_test2 = test[test["done"]==1]
final_weights = np.array([0,0,0,0,0,0,0,0,0])
for lists in actual_test2[colname]:
    final_weights = final_weights + lists


print (features)
print ("corresponding weights")
print (final_weights)

# x   y   hour    week    month   year    accuracy
# fw = [500., 1000., 4., 3., 2., 10., 10.] #feature weights (black magic here)
# [5.11476705   11.07085273   36.80707125   40.54821122   21.04588442  92.50176326     1.37475328]