from tsmixer_model import TSMixer
from local_datasets import SnoozeDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import datetime
import sys

dataset_path = sys.argv[1]

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_datasets(path, ratio=0.8):
  df = pd.read_csv(path).query('Duration > 0.2 and Duration < 40')
  df.logdate = df.logdate.astype('datetime64[ns]')
  df.loc[:, 'day'] = df.logdate.apply(lambda d: datetime.date(d.year, d.month, d.day))
  days = df.day.drop_duplicates().sort_values().values
  date_sep = days[int(len(days) * ratio)]

  train_ds = SnoozeDataset(df[df.day < date_sep], dev)
  test_ds = SnoozeDataset(df[df.day >= date_sep], dev)

  return train_ds, test_ds

train_ds, test_ds = create_datasets(dataset_path)

dl_train = DataLoader(train_ds, batch_size=128, shuffle=True)
dl_test = DataLoader(test_ds, batch_size=len(test_ds), shuffle=True)

model = TSMixer(input_size=(15, 2), hidden_size=64, output_size=1, blocks_n=4)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  for batch, (X, y) in enumerate(dataloader):
      # Compute prediction and loss
      pred = model(X)
      loss = loss_fn(pred, y)

      # Backpropagation
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if batch % 100 == 0:
          loss, current = loss.item(), (batch + 1) * len(X)
          print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
  model.eval()
  size = len(dataloader.dataset)
  num_batches = len(dataloader)
  test_loss, correct = 0, 0

  with torch.no_grad():
      for X, y in dataloader:
          pred = model(X)
          test_loss += loss_fn(pred, y).item()
          correct += (pred.argmax(1) == y).type(torch.float).sum().item()

  test_loss /= num_batches
  correct /= size
  print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(dl_train, model, loss_fn, optimizer)
    test_loop(dl_test, model, loss_fn)
print("Done!")
