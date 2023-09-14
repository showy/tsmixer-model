from tsmixer_model import TSMixer
from snooze_ds_seq import SnoozeDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import datetime


def create_datasets(path, ratio=0.8):
  df = pd.read_csv(path).query('Duration > 0.2 and Duration < 40')
  df.logdate = df.logdate.astype('datetime64[ns]')
  df.loc[:, 'day'] = df.logdate.apply(lambda d: datetime.date(d.year, d.month, d.day))
  days = df.day.drop_duplicates().sort_values().values
  date_sep = days[int(len(days) * ratio)]

  train_ds = SnoozeDataset(df[df.day < date_sep])
  test_ds = SnoozeDataset(df[df.day >= date_sep])

  return train_ds, test_ds

train_ds, test_ds = create_datasets('/home/sah0337/Downloads/dataset_prepared.csv')

dl_train = DataLoader(train_ds, batch_size=128, shuffle=True)
dl_test = DataLoader(test_ds, batch_size=len(test_ds), shuffle=True)

model = TSMixer(input_size=(15, 2), hidden_size=64, output_size=1, blocks_n=4)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

for epoch in range(100):
  for x, y in dl_train:
    optimizer.zero_grad()
    y_pred = model(x)
    loss = loss_fn(y_pred, y)
    loss.backward()
    optimizer.step()

  for x, y in dl_test:
    y_pred = model(x)
    testloss = loss_fn(y_pred, y)

  print(f'Epoch {epoch} train loss: {loss.item()}')
  print(f'Epoch {epoch} test loss: {testloss.item()}')
