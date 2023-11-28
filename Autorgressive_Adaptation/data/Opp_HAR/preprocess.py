import torch
import numpy as np

# Define file paths
input_file_path = 'S2-ADL5.dat'
output_file_path = 'test_b.pt'

# Read data from the .dat file
data = np.loadtxt(input_file_path)

# Extract content and label columns
content = torch.tensor(data[:, :243], dtype=torch.float32)
labels = torch.tensor(data[:, 243:], dtype=torch.float32)

# Save the content and labels into a new .pt file
torch.save({'samples': content, 'labels': labels}, output_file_path)
