import torch
import numpy as np

# Define file paths
input_dir = '../OpportunityUCIDataset/dataset/'
names = ['a','b','c','d']
domain_map = { 'a':1 , 'b':2, 'c':3, 'd':4 }
data_map = {'train':1 , 'val':4, 'test':5 }
for name in names:
    for sec in data_map.keys():
        input_file_path = input_dir+'S'+str(domain_map[name])+'-ADL'+str(data_map[sec])+'.dat'
        output_file_path = sec+'_'+name+'.pt'

        # Read data from the .dat file
        data = np.loadtxt(input_file_path)

        # Extract content and label columns
        content = torch.tensor(data[:, :243], dtype=torch.float32)
        labels = torch.tensor(data[:, 243:], dtype=torch.float32)

        # Save the content and labels into a new .pt file
        torch.save({'samples': content, 'labels': labels}, output_file_path)
