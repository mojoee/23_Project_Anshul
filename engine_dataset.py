import numpy as np
import torch as T
import os
from torch.utils.data import Dataset, DataLoader

class EngineDataset(T.utils.data.Dataset):
  # crank_angle  pressure   time  
  #     -2         2.49      0 
  #     -1         2.56    0.0001  
  # crank_angle -->ca
  # pressure --> cp
  # 

  def __init__(self, src_file, device, m_rows=None):
    all_xy = np.genfromtxt(src_file, max_rows=m_rows,
      usecols=[0,2], delimiter=",",
      comments="#", dtype=np.float32)


    tmp_x = all_xy[:,[0]]
    tmp_y = all_xy[:,[1]]  # 2-D

    self.x_data = T.tensor(tmp_x, \
      dtype=T.float32).to(device)
    self.y_data = T.tensor(tmp_y, \
      dtype=T.float32).to(device)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    angle = self.x_data[idx,:]  # or just [idx]
    pressure = self.y_data[idx,:] 
    return (angle, pressure)       # tuple of matrices


def create_dataset():
    device = T.device("cuda")
    merged_data=os.path.join("data", "merged_data.csv")
    dataset= EngineDataset(merged_data, device)
    return dataset

def load_dataset(test_sen=None, test_split=0.2, val_split=0.2):
 
    dataset=create_dataset() 
    dataset_size=len(dataset)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - test_size -val_size 
    train_data, valid_data, test_data = T.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=T.Generator().manual_seed(42))
    train_iter, valid_iter, test_iter = DataLoader((train_data, valid_data, test_data), batch_size=4,
                            shuffle=True, num_workers=0)

    return train_iter, valid_iter, test_iter
