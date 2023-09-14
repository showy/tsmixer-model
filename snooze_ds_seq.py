import torch
import pandas as pd
from torch.utils.data import Dataset
import datetime

dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class SnoozeDataset(Dataset):
  FEATURES = [
    'DayOfWeek',
    'DayTime',
  ] + [ f"state_{i}_day_before" for i in range(1, 15) ] + [ f"Duration{i}_day_before" for i in range(1, 15)]

  TARGET = 'Duration'

  def __init__(self, df):
    self.df = df
    
  def __len__(self):
    return len(self.df)
  
  def __getitem__(self, idx):
    y = torch.Tensor([self.df.iloc[idx][SnoozeDataset.TARGET]], device=dev)
    x = torch.Tensor(self.df.iloc[idx][SnoozeDataset.FEATURES].values.reshape(15, 2).tolist(), device=dev)
    return x, y
  