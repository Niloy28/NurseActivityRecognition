import numpy as np
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
from dtaidistance import dtw
import matplotlib.pyplot as plt
import math
import csv
import array
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from numpy import savetxt

dataset = pd.read_csv("../dataset/Training/Lab/bigact_raw_lab_acc.csv")
label = pd.read_csv("../dataset/Training/Lab/labels_lab_2users.csv")

#l = label['start'].value_counts()
#data_per_user = dataset.shape[0] / label.shape[0]

modified_dataset = dataset.iloc[:, [2,3,4]].values
#modified_dataset = np.reshape(modified_dataset,(420,-1))
Y = label.iloc[:-1,1].values

#label = label.iloc[:,:].values
#label['act_id'].value_counts()

datetime = dataset[['datetime','user_id']].values
datetime_array = np.zeros((datetime.shape[0], 6))
label_datetime_start = label['start'].values
label_datetime_finish = label['finish'].values
label_user_id = label['user_id'].values
label_datetime = np.concatenate((label_datetime_start.reshape(-1,1), label_datetime_finish.reshape(-1,1),label_user_id.reshape(-1,1)), axis = 1)
label_datetime_array = np.zeros((label.shape[0], 6))

for i in range(len(datetime)):
    year = datetime[i,0][0:4]
    month = datetime[i,0][5:7]
    date = datetime[i,0][8:10]
    hour = datetime[i,0][11:13]
    minute = datetime[i,0][14:16]
    second = datetime[i,0][17:23]
    user_id = datetime[i,1]
    hour_minute = int(str(hour) + str(minute))
    datetime_array[i][0] = year
    datetime_array[i][1] = month
    datetime_array[i][2] = date
    datetime_array[i][3] = hour_minute
    #datetime_array[i][4] = minute
    datetime_array[i][4] = second
    datetime_array[i][5] = user_id

#d = pd.DataFrame(datetime_array)
#d = datetime_array[:,3]
#d[3].value_counts()

for i in range(len(label_datetime)):
    month = label_datetime[i,0][0]
    date = label_datetime[i,0][2:4]
    year = label_datetime[i,0][5:9]
    hour_start = label_datetime[i,0][10:12]
    hour_stop = label_datetime[i,1][10:12]
    minute_start = label_datetime[i,0][13:15]
    minute_stop = label_datetime[i,1][13:15]
    user_id = label_datetime[i,2]
    hour_minute_start = int(str(hour_start) + str(minute_start))
    hour_minute_stop = int(str(hour_stop) + str(minute_stop))
    label_datetime_array[i][0] = year
    label_datetime_array[i][1] = month
    label_datetime_array[i][2] = date
    label_datetime_array[i][3] = hour_minute_start
    label_datetime_array[i][4] = hour_minute_stop
    label_datetime_array[i][5] = user_id
    #label_datetime_array[i][4] = minute

#d = pd.DataFrame(label_datetime_array)
#l = label_datetime_array[:,-1]
#d[0].value_counts()

label_list = []
#datetime_index = np.zeros((len(dataset),7))
for i in range(len(label_datetime)-1):
    data_list = []
    year = label_datetime_array[i][0]
    month = label_datetime_array[i][1]
    date = label_datetime_array[i][2]
    hour_minute_start = label_datetime_array[i][3]
    hour_minute_stop = label_datetime_array[i][4]
    user_id = label_datetime_array[i][5]
    #minute_start = label_datetime_array[i][4]
    #minute_stop = label_datetime_array[i+1][4]
    for j in range(len(datetime)):
        if datetime_array[j][0] == year and datetime_array[j][1] == month and datetime_array[j][2] == date and (datetime_array[j][3] >= hour_minute_start and datetime_array[j][3] <= hour_minute_stop) and datetime_array[j][5] == user_id:

            data_list.append(j)
            #datetime_index[j][0] = datetime_array[j][0]  #year
            #datetime_index[j][1] = datetime_array[j][1]  #month
            #datetime_index[j][2] = datetime_array[j][2]  #day
            #datetime_index[j][3] = datetime_array[j][3]  #hour
            #datetime_index[j][4] = datetime_array[j][4]  #minute
            #datetime_index[j][5] = datetime_array[j][5]  #second
            #datetime_index[j][6] = j  #index

    label_list.append(data_list)

label_list = np.array(label_list)
label_list_ = np.concatenate((label_list.reshape(-1,1), Y.reshape(-1,1)), axis = 1)
count = 0
for i in range(len(label_list)):
    index_list = label_list_[i][0]
    if len(index_list) != 0:
        for j in index_list:
            count += 1

X = np.zeros((count,4))
count = 0
for i in range(len(label_list)):
    index_list = label_list_[i][0]
    if len(index_list) != 0:
        for j in index_list:
            X[count][0] = modified_dataset[j][0]
            X[count][1] = modified_dataset[j][1]
            X[count][2] = modified_dataset[j][2]
            X[count][3] = label_list_[i][1]
            count = count + 1
"""
for i in range(len(X)):
    if X[i][1] == 0:
        X = np.delete(X,i,0)
"""
savetxt('sample_data.csv', X, delimiter=',')

#X_ = pd.DataFrame(X)
#X_[3].value_counts()
#dataframe_label = pd.DataFrame(label_list_)
#dataframe_label[1].value_counts()
"""
for i in range(len(datetime)):
    if datetime_array[i][0] == 2018 and  datetime_array[i][1] == 7 and datetime_array[i][2] == 25 and (datetime_array[i][3] >= 1226 and datetime_array[i][3] < 1227):
        print(i)
"""