import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.utils import shuffle
from torch.utils.data import DataLoader


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

def getAllDadaSet(path):

    data_set = pd.read_csv(path, index_col='id')

    print(data_set.shape)

    x_data = data_set.loc[:, data_set.columns != 'address']
    y_data = data_set.loc[:, data_set.columns == 'address']

    # 归一化
    x_data = (x_data-x_data.min()) / (x_data.max() - x_data.min())

    data_set = pd.concat([x_data, y_data], axis=1)

    #打乱数据
    data_set = shuffle(data_set)

    train_data_size = int(data_set.shape[0]*0.8)

    trainData = data_set.iloc[:train_data_size, ]
    testData = data_set.iloc[train_data_size:, ]

    trainData = shuffle(trainData)
    testData = shuffle(testData)

    x_train = trainData.loc[:, trainData.columns != 'address']
    y_train = trainData.loc[:, trainData.columns == 'address']

    x_test = testData.loc[:, testData.columns != 'address']
    y_test = testData.loc[:, testData.columns == 'address']


    print("x_train shape:", x_train.shape)
    print("y_train shape:", y_train.shape)
    print("x_test shape:", x_test.shape)
    print("y_test shape:", y_test.shape)

    x_train.astype(np.float32)
    x_test.astype(np.float32)

    x_train = x_train.fillna(0)
    x_test = x_test.fillna(0)

    x_train = torch.tensor(data=x_train.values, dtype=torch.float32)
    y_train = torch.tensor(data=y_train.values).type(torch.LongTensor)

    x_test = torch.tensor(data=x_test.values, dtype=torch.float32)
    y_test = torch.tensor(data=y_test.values).type(torch.LongTensor)

    train_data = GetLoader(x_train, y_train)
    test_data = GetLoader(x_test, y_test)

    return train_data, test_data



def getAllTargetDataSet(batch_size):
    path = r'../dataset/lcad.csv'
    train_data, test_data= getAllDadaSet(path)
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    return train_loader, test_loader






