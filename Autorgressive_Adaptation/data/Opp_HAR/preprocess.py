import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as fun
import argparse
import pandas as pd

DEBUG = 0
# dirty_col = set()

class TimeSeriesDataset(Dataset):
    def __init__(self, file_paths, window_size, overlap):
        self.data = [self.load_data(file_path) for file_path in file_paths]
        self.window_size = window_size
        self.overlap = overlap
        self.content, self.labels = self.process_data()


    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]
    
    def get_remove_index(self):
        return [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 134, 135, 136, 137, 138, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 163, 189, 190, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 219, 220, 221, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242]

    def load_data(self, file_path):
        # Load data from the .dat file using np.loadtxt
        data = np.loadtxt(file_path)
        content = data[:,:243]
        label = data[:,243]
        return (content, label)

    def nan_list(df):
        nulls = df.isna().to_numpy()
        zeros = np.zeros(nulls.shape)
        zeros[nulls] = 1
        return list(np.sum(zeros,axis=0))

    def clean_data(self):
        # global dirty_col
        datas = []
        for content, label in self.data:
            df = pd.DataFrame(content)
            # print(self.remove_index)
            df.drop(columns=self.get_remove_index(), inplace=True)
            # print(len(df))
            # print("Before interpolate")
            # list1 = TimeSeriesDataset.nan_list(df)
            # dirty_columns = set([ind for ind, val in zip(range(len(list1)), list1) if val>0.152*len(df)])
            # dirty_col = dirty_col.union(dirty_columns)
            df.interpolate('linear', inplace=True)
            # print(TimeSeriesDataset.nan_list(df.iloc[:,:113]))
            # print(df.shape)
            df.dropna(axis=1,inplace=True)  
            # print(df.shape)         
            datas.append((df.to_numpy(), label))
        self.data = datas
            
    def get_label(self, data, idx):
        check_map = {0:0,1:1,2:2,4:3,5:3}
        return torch.tensor(check_map[data[idx+self.window_size-1:idx + self.window_size].item()], dtype=torch.int64)

    def process_data(self):
        self.clean_data()
        processed_content = []
        processed_labels = []
        for (data_con, data_label) in self.data:
            for idx in range(0, len(data_con) - self.window_size + 1, int(self.window_size * (1 - self.overlap))):
                content_data = torch.tensor(data_con[idx:idx + self.window_size, :113], dtype=torch.float32)
                labels_data = self.get_label(data_label,idx)
                # content_tensor = torch.tensor(content_data, dtype=torch.float32)
                # labels_tensor = torch.tensor(labels_data, dtype=torch.float32)
                processed_content.append(content_data)
                processed_labels.append(labels_data)

        # Stack the 2D tensors along a new dimension to create a 3D tensor
        processed_content = torch.stack(processed_content, dim=0)
        processed_labels = torch.stack(processed_labels, dim=0)
        return processed_content, processed_labels

def main():
    parser = argparse.ArgumentParser(description="Process time series data with sliding windows.")
    parser.add_argument("--window_size", type=int, default=128, help="Size of the sliding window (default: 128)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap percentage between windows (default: 0.5)")
    
    args = parser.parse_args()

    input_dir = '../OpportunityUCIDataset/dataset/'
    names = ['a','b','c','d']
    domain_map = { 'a':1 , 'b':2, 'c':3, 'd':4 }
    data_map = {'train': [1,2,3] , 'val':[4], 'test':[5] }
    for name in names:
        for sec in data_map.keys():
            #Input file paths
            input_file_paths = list()
            for id in data_map[sec]:
                input_file_paths.append(input_dir+'S'+str(domain_map[name])+'-ADL'+str(id)+'.dat')
            output_file_path = sec+'_'+name+'.pt'

            #Creat the datasets
            dataset = TimeSeriesDataset(input_file_paths, args.window_size, args.overlap)

            if (DEBUG):
                break
            # Save the processed data as a .pt file
            torch.save({'samples':dataset.content, 'labels':dataset.labels}, output_file_path)
            print(dataset.content.size())
            print(dataset.labels.size())
        if (DEBUG):
            break
    # print(dirty_col)
    # print(len(list(dirty_col)))
if __name__ == "__main__":
    main()