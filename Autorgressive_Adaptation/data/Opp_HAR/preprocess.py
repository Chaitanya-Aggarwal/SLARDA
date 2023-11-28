import torch
import numpy as np
from torch.utils.data import Dataset
import argparse

class TimeSeriesDataset(Dataset):
    def __init__(self, file_path, window_size, overlap):
        self.data = self.load_data(file_path)
        self.window_size = window_size
        self.overlap = overlap
        self.content, self.labels = self.process_data()

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        return self.processed_data[idx]

    def load_data(self, file_path):
        # Load data from the .dat file using np.loadtxt
        data = np.loadtxt(file_path)

        return data

    def process_data(self):
        processed_data = []
        processed_content = []
        processed_labels = []
        for idx in range(0, len(self.data) - self.window_size + 1, int(self.window_size * (1 - self.overlap))):
            window_data = self.data[idx:idx + self.window_size]
            content_data = torch.tensor(self.data[idx:idx + self.window_size, :113], dtype=torch.float32)
            labels_data = torch.tensor(self.data[idx:idx + self.window_size, 243], dtype=torch.float32)
            window_tensor = torch.tensor(window_data, dtype=torch.float32)
            content_tensor = torch.tensor(content_data, dtype=torch.float32)
            labels_tensor = torch.tensor(labels_data, dtype=torch.float32)
            processed_data.append(window_tensor)
            processed_content.append(content_tensor)
            processed_labels.append(labels_tensor)

        # Stack the 2D tensors along a new dimension to create a 3D tensor
        processed_data = torch.stack(processed_data, dim=0)
        processed_content = torch.stack(processed_content, dim=0)
        processed_labels = torch.stack(processed_labels, dim=0)

        return processed_content, processed_data

def main():
    parser = argparse.ArgumentParser(description="Process time series data with sliding windows.")
    parser.add_argument("--window_size", type=int, default=128, help="Size of the sliding window (default: 128)")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap percentage between windows (default: 0.5)")
    
    args = parser.parse_args()

    input_dir = '../OpportunityUCIDataset/dataset/'
    names = ['a','b','c','d']
    domain_map = { 'a':1 , 'b':2, 'c':3, 'd':4 }
    data_map = {'train':1 , 'val':4, 'test':5 }
    for name in names:
        for sec in data_map.keys():
            input_file_path = input_dir+'S'+str(domain_map[name])+'-ADL'+str(data_map[sec])+'.dat'
            output_file_path = sec+'_'+name+'.pt'
        
            # Create the dataset with command-line arguments
            dataset = TimeSeriesDataset(input_file_path, args.window_size, args.overlap)

            # Save the processed data as a .pt file
            torch.save({'samples':dataset.content, 'labels':dataset.labels}, output_file_path)

if __name__ == "__main__":
    main()


# import torch
# import numpy as np
# import sys

# # Define file paths
# input_dir = '../OpportunityUCIDataset/dataset/'
# names = ['a','b','c','d']
# domain_map = { 'a':1 , 'b':2, 'c':3, 'd':4 }
# data_map = {'train':1 , 'val':4, 'test':5 }
# for name in names:
#     for sec in data_map.keys():
#         input_file_path = input_dir+'S'+str(domain_map[name])+'-ADL'+str(data_map[sec])+'.dat'
#         output_file_path = sec+'_'+name+'.pt'
    
#         # Read data from the .dat file
#         data = np.loadtxt(input_file_path)

#         # Extract content and label columns
#         content = torch.tensor(data[:, :113], dtype=torch.float32)
#         labels = torch.tensor(data[:, 243], dtype=torch.float32)

#         # Save the content and labels into a new .pt file
#         torch.save({'samples': content, 'labels': labels}, output_file_path)

# #LALALALA