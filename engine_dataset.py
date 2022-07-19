import numpy as np
import torch as T
import os
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as TF
from sklearn.preprocessing import MinMaxScaler
from torch.autograd import Variable
import pandas as pd

class EngineDataset(T.utils.data.Dataset):
  # crank_angle  pressure   time  
  #     -2         2.49      0 
  #     -1         2.56    0.0001  
  # crank_angle -->ca
  # pressure --> cp
  # 

  def __init__(self, src_file, device, config, m_rows=None):
    all_xy = np.genfromtxt(src_file, max_rows=m_rows,
      usecols=[0,2], delimiter=",",
      comments="#", dtype=np.float32)

    tmp_x = all_xy[:,[0]]
    tmp_y = all_xy[:,[1]]  # 2-D

    self.transform = TF.Compose([TF.ToTensor()]) 
    self.create_protocol(config)

  def __len__(self):
    return len(self.x_data)

  def __getitem__(self, idx):
    angle = self.x_data[idx,:]  # or just [idx]
    pressure = self.y_data[idx,:] 
    return angle, pressure      # tuple of matrices

  def create_protocol(self, config):
    training_set = pd.read_csv('./sample/CylinderPressure.csv')
    stepsize=config.data_stepsize
    training_set = training_set.iloc[::stepsize,0:1].values
    self.sc = MinMaxScaler()
    training_data = self.sc.fit_transform(training_set)

    seq_length = config.seq_length
    x, y = sliding_windows(training_data, seq_length)

    self.train_size = int(len(y) * 0.67)
    self.test_size = len(y) - self.train_size

    self.x_data = Variable(T.Tensor(np.array(x)))
    self.y_data = Variable(T.Tensor(np.array(y)))

    self.trainX = Variable(T.Tensor(np.array(x[0:self.train_size])))
    self.trainY = Variable(T.Tensor(np.array(y[0:self.train_size])))

    self.testX = Variable(T.Tensor(np.array(x[self.train_size:len(x)])))
    self.testY = Variable(T.Tensor(np.array(y[self.train_size:len(y)])))

def sliding_windows(data, seq_length):
    x = []
    y = []

    for i in range(len(data)-seq_length-1):
        _x = data[i:(i+seq_length)]
        _y = data[i+seq_length]
        x.append(_x)
        y.append(_y)

    return np.array(x),np.array(y)


def create_dataset(config):
  device = T.device("cuda")
  merged_data=os.path.join("data", "merged_data.csv")
  dataset= EngineDataset(merged_data, device, config)
  return dataset

def load_dataset(config, test_sen=None, test_split=0.2, val_split=0.2):
 
  dataset=create_dataset(config) 
  dataset_size=len(dataset)
  test_size = int(test_split * dataset_size)
  val_size = int(val_split * dataset_size)
  train_size = dataset_size - test_size -val_size 
  shuffled=False
  if shuffled:
    train_data, valid_data, test_data = T.utils.data.random_split(dataset, [train_size, val_size, test_size], generator=T.Generator().manual_seed(42))
    train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    valid_iter = DataLoader(valid_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(test_data, batch_size=test_size, shuffle=False, num_workers=0)
  else:
    train_data=T.utils.data.Subset(dataset, range(train_size))
    val_data = T.utils.data.Subset(dataset, range(train_size, train_size + val_size))
    test_data = T.utils.data.Subset(dataset, range(train_size+val_size, train_size + val_size+ test_size))
    
  train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
  valid_iter = DataLoader(val_data, batch_size=config.batch_size, shuffle=True, num_workers=0)
  test_iter = DataLoader(test_data, batch_size=test_size, shuffle=False, num_workers=0)

  return train_iter, valid_iter, test_iter, dataset
