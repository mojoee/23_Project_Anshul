import numpy as np
import torch as T
import os

class HouseDataset(T.utils.data.Dataset):
  # crank_angle  pressure   time  
  #     -2         2.49      0 
  #     -1         2.56    0.0001  
  # crank_angle -->ca
  # pressure --> cp
  # 

  def __init__(self, src_file, m_rows=None):
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
    preds = self.x_data[idx,:]  # or just [idx]
    price = self.y_data[idx,:] 
    return (preds, price)       # tuple of matrices

if __name__=="__main__":
    device = T.device("cuda")
    input=os.path.join("data", "CrankAnglePosition_small.csv")
    target=os.path.join("data", "CylinderPressure_small.csv")
    merged_data=os.path.join("data", "merged_data.csv")
    house_data= HouseDataset(merged_data)
    print(house_data)