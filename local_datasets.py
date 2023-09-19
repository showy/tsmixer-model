import torch
import pandas as pd
from torch.utils.data import Dataset
import datetime

class SnoozeDataset(Dataset):
  FEATURES = [
    'DayOfWeek',
    'DayTime',
  ] + [ f"state_{i}_day_before" for i in range(1, 15) ] + [ f"Duration{i}_day_before" for i in range(1, 15)]

  TARGET = 'Duration'

  def __init__(self, df, device='cpu'):
    self.df = df
    self.x = torch.Tensor(self.df[SnoozeDataset.FEATURES].values.reshape(-1, 15, 2).tolist(), device=device)
    self.y = torch.Tensor(self.df[SnoozeDataset.TARGET].values, device=device)

  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    x = self.x[idx]
    y = self.y[idx]
    return x, y
  

class ElecDataset(Dataset):
  def __init__(self,feature,target):
    self.feature = torch.Tensor(feature)
    self.target = torch.Tensor(target)
  
  def __len__(self):
    return len(self.feature)
  
  def __getitem__(self,idx):
    item = self.feature[idx]
    label = self.target[idx]
    
    return item,label
