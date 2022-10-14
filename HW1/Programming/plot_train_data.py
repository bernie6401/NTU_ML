from statistics import mean
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import csv

'''
Load data
'''
train_data = pd.read_csv("./dataset/train.csv")
train_data = train_data.values
train_data = np.transpose(np.array(np.float64(train_data)))




'''
Counting invalid data
'''
def count_invalid_data(train_data):
    count = 0
    j = 14
    for i in range(len(train_data[j])):
        if train_data[j][i] > 70:  #train_data[1][0] > 1.2 or train_data[2][0] > 40 or train_data[3][0] > 50 or train_data[4][0] > 80 or train_data[6][0] > 200 or train_data[14][0] > 70
            count+=1

    print(count)
    return count

'''
Visulize training data
'''
def visul_data(data, path, norm_type=0):
    dic_data_title = {
        0 : 'AMB_TEMP', 1 : 'CO', 2 : 'NO', 3 : 'NO2', 4 : 'NOx',
        5 : 'O3', 6 : 'PM10', 7 : 'WS_HR', 8 : 'RAINFALL', 9 : 'RH',
        10 : 'SO2', 11 : 'WD_HR', 12 : 'WIND_DIREC', 13 : 'WIND_SPEED', 14 : 'PM2.5'
    }
    dic_norm_type = {
        0:'_zscore_norm.png',
        1:'_maxmin_norm.png',
        2:'_maxabs_norm.png',
        3:'_robust_norm.png'
    }

    for i in range(15):
        j = np.arange(len(data[i]))
        plt.scatter(j, data[i])
        plt.xlabel('i-th data')
        plt.ylabel(dic_data_title[i])
        if norm_type == -1:
            plt.savefig(path + dic_data_title[i] + '.png')
        else:
            plt.savefig(path + dic_data_title[i] + dic_norm_type[norm_type])
        plt.show()

'''
Normalize training data
'''
def norm_data(data, norm_case, mean_arr=0, std_arr=0):
    dic_data_round = {
        0 : 1, 1 : 2, 2 : 1, 3 : 1, 4 : 1,
        5 : 1, 6 : 0, 7 : 1, 8 : 1, 9 : 0, 
        10 : 1, 11 : 0, 12 : 0, 13 : 1, 14 : 0
    }

    # Z-Score
    if norm_case == 0:
        train_data = data
        mean_arr = np.zeros(15)
        std_arr = np.zeros(15)

        for i in range(15):
            # Compute mean
            mean_temp = sum(train_data[i])
            mean_arr[i] = mean_temp / float(len(train_data[i]))
        
            # Compute std
            std_temp = 0
            for j in range(len(train_data[i])):
                std_temp += (train_data[i][j] - mean_arr[i])**2
            std_arr[i] = (std_temp / float((len(train_data[i] - 1))))**0.5

            # Create normalize data
            train_data[i] -= mean_arr[i]
            train_data[i] = np.round(train_data[i] / std_arr[i], dic_data_round[i]+1)
        return train_data, mean_arr, std_arr

    # Max-Min
    elif norm_case == 1:
        train_data = data
        for i in range(15):
            max_temp = max(train_data[i])
            min_temp = min(train_data[i])
            train_data[i] = np.round((data[i] - min_temp) / (max_temp - min_temp), dic_data_round[i]+1)
        return train_data, max_temp, min_temp
    
    # MaxAbs
    elif norm_case == 2:
        train_data = data
        for i in range(15):
            maxabs_temp = abs(max(train_data[i]))
            train_data[i] = np.round(train_data[i] / maxabs_temp, dic_data_round[i]+1)
        return train_data, maxabs_temp

    # RobustScaler
    elif norm_case == 3:
        train_data = data

        return train_data

    # Normalize Testing Data
    elif norm_case == 4:
        test_data = data
        for i in range(15):
            test_data[i] -= mean_arr[i]
            test_data[i] = np.round(test_data[i] / std_arr[i], dic_data_round[i]+1)
        return test_data

    else:
        print('No define the normalize method, please check again')
        return data

'''
Filter valid data
'''
def valid(x):
    # Total unvalid datas is 35
    global count
    dic = {
        'AMB_TEMP':0, 'CO':1, 'NO':2, 'NO2':3, 'NOx':4, 'O3':5,
        'PM10':6, 'WS_HR':7, 'RAINFALL':8, 'RH':9, 'SO2':10,
        'WD_HR':11, 'WIND_DIREC':12, 'WIND_SPEED':13, 'PM2.5':14}
    for i in range(len(x[0])):
        if x[dic['CO']][i] > 1.4 or x[dic['NO']][i] > 40 or x[dic['NO2']][i] > 55 or \
            x[dic['NOx']][i] > 75 or x[dic['O3']][i] > 120 or x[dic['PM10']][i] > 150 or \
            x[dic['RAINFALL']][i] > 30 or x[dic['SO2']][i] > 20 or x[dic['PM2.5']][i] > 65:
            count.append(i)
    count.reverse()
    for i in range(len(count)):
        for j in range(15):
            del x[j][count[i]]
    return np.array(x)


'''
Testing Data
'''
# def test_data(test_data):
#     # test_x = np.array(test_data)
#     with open('./model/weight_best.pickle', 'rb') as f:
#         w = pickle.load(f)
#     with open('./model/bias_best.pickle', 'rb') as f:
#         bias = pickle.load(f)
    
#     with open('./testing_result/my_sol_best.csv', 'w', newline='') as csvf:
#         # 建立 CSV 檔寫入器
#         writer = csv.writer(csvf)
#         writer.writerow(['Id','Predicted'])

#         print(test_x.shape) 
#         for i in range(int(test_x.shape[0])):
#             # Prediction of linear regression 
#             prediction = (np.dot(np.reshape(w,-1),test_x[i]) + bias)[0]
#             writer.writerow([i, prediction] )

def parse2test(data, feats):
    x = []
    for i in range(90):
        x_tmp = data[feats, 8*i:8*i+8]
        x.append(x_tmp.reshape(-1,))

    # x.shape: (n, 15, 8)
    x = np.array(x)
    return x


test_data = pd.read_csv('./dataset/test.csv')
test_data = test_data.values
test_data = np.transpose(np.array(np.float64(test_data)))
# feats = [1, 2, 3, 4, 6, 7, 8, 9, 13, 14]
# test_x = parse2test(test_data, feats)
# with open('./model/weight_best.pickle', 'rb') as f:
#     w = pickle.load(f)
# with open('./model/bias_best.pickle', 'rb') as f:
#     bias = pickle.load(f)

# with open('./testing_result/my_sol_best.csv', 'w', newline='') as csvf:
#     # 建立 CSV 檔寫入器
#     writer = csv.writer(csvf)
#     writer.writerow(['Id','Predicted'])

#     print(test_x.shape) 
#     for i in range(int(test_x.shape[0])):
#         # Prediction of linear regression 
#         prediction = (np.dot(np.reshape(w,-1),test_x[i]) + bias)[0]
#         writer.writerow([i, prediction] )

visul_data(test_data, './test_data_img/', norm_type=-1)