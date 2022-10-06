import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

data = pd.read_csv("./train.csv")
data = data.values
train_data = np.transpose(np.array(np.float64(data)))

# count = 0

# for i in range(len(train_data[-1])):
#     if train_data[-1][i] >= 50:
#         count+=1

# print(count)

dic_data_title = {
    0 : 'AMB_TEMP',
    1 : 'CO',
    2 : 'NO',
    3 : 'NO2',
    4 : 'NOx',
    5 : 'O3',
    6 : 'PM10',
    7 : 'WS_HR',
    8 : 'RAINFALL',
    9 : 'RH',
    10 : 'SO2',
    11 : 'WD_HR',
    12 : 'WIND_DIREC',
    13 : 'WIND_SPEED',
    14 : 'PM2.5'
}

for i in range(15):
    j = np.arange(len(train_data[i]))
    plt.scatter(j, train_data[i])
    plt.xlabel('i-th data')
    plt.ylabel(dic_data_title[i])
    plt.savefig('./train_data_img/' + dic_data_title[i] + '.png')
    plt.show()