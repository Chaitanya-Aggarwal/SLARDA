from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import math
import os
import torch

activities = {1: 'stand',
              2: 'walk',
              4: 'sit',
              5: 'lie',
              101: 'relaxing',
              102: 'coffee time',
              103: 'early morning',
              104: 'cleanup',
              105: 'sandwich time'
               }


def read_files():
    #pick partial data from dataset
    common_path = "../OpportunityUCIDataset/dataset/"
    list_of_files = [common_path+"S1-ADL1.dat",
                     common_path+"S1-ADL2.dat",
                     common_path+"S1-ADL3.dat",
                     common_path+"S1-ADL4.dat",
                     common_path+"S1-ADL5.dat",
                     common_path+"S2-ADL1.dat",
                     common_path+"S2-ADL2.dat",
                     common_path+"S2-ADL3.dat",
                     common_path+"S2-ADL4.dat",
                     common_path+"S2-ADL5.dat",
                     common_path+"S3-ADL1.dat",
                     common_path+"S3-ADL2.dat",
                     common_path+"S3-ADL3.dat",
                     common_path+"S3-ADL4.dat",
                     common_path+"S3-ADL5.dat",
                     common_path+"S4-ADL1.dat",
                     common_path+"S4-ADL2.dat",
                     common_path+"S4-ADL3.dat",
                     common_path+"S4-ADL4.dat",
                     common_path+"S4-ADL5.dat",
                    ]
    
    list_of_drill = ['./dataset/S1-Drill.dat',
                     './dataset/S2-Drill.dat',
                     './dataset/S3-Drill.dat',
                     './dataset/S4-Drill.dat',
                     ]
    col_names = []

    with open('./col_names','r') as f:# a file with all column names was created
        lines = f.read().splitlines()
        for line in lines:
            col_names.append(line)
    
    dataCollection = pd.DataFrame()
    for i, file in enumerate(list_of_files):
        # print(file," is reading...")
        procData = pd.read_table(file, header=None, sep='\s+')
        procData.columns = col_names
        procData['file_index'] = i # put the file index at the end of the row
        dataCollection = dataCollection._append(procData, ignore_index=True)       
    dataCollection.reset_index(drop=True, inplace=True)
    
    return dataCollection


def dataCleaning(dataCollection):
    dataCollection = dataCollection.loc[:,dataCollection.isnull().mean()< 0.1] #drop the columns which has NaN over 10%
    dataCollection = dataCollection.drop(['MILLISEC', 'LL_Left_Arm','LL_Left_Arm_Object','LL_Right_Arm','LL_Right_Arm_Object', 'ML_Both_Arms'],
                                        axis = 1)  # removal of columns not related, may include others.
    
    dataCollection = dataCollection.apply(pd.to_numeric, errors = 'coerce') #removal of non numeric data in cells
    
    # print(dataCollection.isna().sum().sum())#count all NaN 
    # print(dataCollection.shape)
    dataCollection = dataCollection.interpolate() 
    # print(dataCollection.isna().sum().sum())#count all NaN 
    # print("data cleaned!")
    return dataCollection

def reset_label(dataCollection, locomotion): 
    # Convert original labels {1, 2, 4, 5, 101, 102, 103, 104, 105} to new labels. 
    mapping = {1:1, 2:2, 5:0, 4:3, 101: 0, 102:1, 103:2, 104:3, 105:4} # old activity id to new activity Id 
    if locomotion: #new labels [0,1,2,3]
        for i in [5,4]: # reset ids in Locomotion column
            dataCollection.loc[dataCollection.Locomotion == i, 'Locomotion'] = mapping[i]
    else: # reset the high level activities ; new labels [0,1,2,3,4]
        for j in [101,102,103,104,105]:# reset ids in HL_activity column
            dataCollection.loc[dataCollection.HL_Activity == j, 'HL_Activity'] = mapping[j]
    return dataCollection

def segment_locomotion(dataCollection, window_size): # segment the data and create a dataset with locomotion classes as labels
    #remove locomotions with 0
    dataCollection = dataCollection.drop(dataCollection[dataCollection.Locomotion == 0].index)
    # reset labels
    dataCollection= reset_label(dataCollection,True)
    #print(dataCollection.columns)
    loco_i = dataCollection.columns.get_loc("Locomotion")
    #convert the data frame to numpy array
    data = dataCollection.to_numpy()
    #segment the data
    n = len(data)
    X = []
    y = []
    file_ind = []
    start = 0
    end = 0
    while start + window_size - 1 < n:
        end = start + window_size-1
        if data[start][loco_i] == data[end][loco_i] and data[start][-1] == data[end][-1] : # if the frame contains the same activity and from the file

            # X.append(data[start:(end+1),0:loco_i])
            X.append(data[start:(end+1),0:113])
            y.append(data[start][loco_i])
            file_ind.append(data[start][-1])
            start += window_size//2 # 50% overlap
        else: # if the frame contains different activities or from different objects, find the next start point
            while start + window_size-1 < n:
                if data[start][loco_i] != data[start+1][loco_i]:
                    break
                start += 1
            start += 1
    print(np.asarray(X).shape, np.asarray(y).shape)
    return {'content' : np.asarray(X), 'labels': np.asarray(y,dtype=int), 'file_index': np.asarray(file_ind, dtype=int)}

def split_array(arr, ratios):
    """
    Split a numpy array into three datasets based on given ratios.

    Parameters:
    - arr: numpy array
        The array to be split.
    - ratios: tuple of floats
        Ratios for splitting the array into three parts.

    Returns:
    - tuple of numpy arrays
        Three arrays representing the split datasets.
    """
    total_length = len(arr)
    split_points = np.cumsum(np.multiply(ratios, total_length)).astype(int)
    print(sum(ratios))

    return np.split(arr, split_points[:-1])

if __name__ == "__main__":   
    window_size = 128 
    common_path = "../OpportunityUCIDataset/dataset/"
    files = [[common_path+"S1-ADL1.dat",
            common_path+"S1-ADL2.dat",
            common_path+"S1-ADL3.dat",
            common_path+"S1-ADL4.dat",
            common_path+"S1-ADL5.dat"],
            [common_path+"S2-ADL1.dat",
            common_path+"S2-ADL2.dat",
            common_path+"S2-ADL3.dat",
            common_path+"S2-ADL4.dat",
            common_path+"S2-ADL5.dat",
            ],
            [common_path+"S3-ADL1.dat",
            common_path+"S3-ADL2.dat",
            common_path+"S3-ADL3.dat",
            common_path+"S3-ADL4.dat",
            common_path+"S3-ADL5.dat",
            ],
            [common_path+"S4-ADL1.dat",
            common_path+"S4-ADL2.dat",
            common_path+"S4-ADL3.dat",
            common_path+"S4-ADL4.dat",
            common_path+"S4-ADL5.dat",
            ]]
    df = read_files()
    df = dataCleaning(df)
    data_loco = segment_locomotion(df, window_size)
    ratios = (0.6,0.2,0.2)

    # train, test, val = split_array(np.arange(data_loco['labels'].shape[0]), ratios)
    # print(data_loco['file_index'])
    map_names = {0:'a',1:'b',2:'c',3:'d'}
    for user in range(4):
        # print(data_loco['file_index']//5)
        X = data_loco['content'][data_loco['file_index']//5==user]
        y = data_loco['labels'][data_loco['file_index']//5==user]
        train, test, val = split_array(np.arange(y.shape[0]),ratios)
        train_X, train_Y = torch.Tensor(X[train]), torch.Tensor(y[train])
        val_X, val_Y = torch.Tensor(X[val]), torch.Tensor(y[val])
        test_X, test_Y = torch.Tensor(X[test]), torch.Tensor(y[test])
        torch.save({'samples':train_X, 'labels':train_Y}, "train_"+map_names[user]+".pt")
        print(train_X.size())
        print(train_Y.size())
        torch.save({'samples':val_X, 'labels':val_Y}, "val_"+map_names[user]+".pt")
        print(val_X.size())
        print(val_Y.size())
        torch.save({'samples':test_X, 'labels':test_Y}, "test_"+map_names[user]+".pt")
        print(test_X.size())
        print(test_Y.size())