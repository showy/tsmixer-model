from tsmixer_model import TSMixer
from cnn import CNN_ForecastNet
from local_datasets import ElecDataset
import torch
from torch.utils.data import DataLoader
import pandas as pd
import datetime
import sys
import numpy as np
import torchsummary

dataset_path = sys.argv[1]

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def create_datasets(path, ratio=0.8):
  df = pd.read_csv(path)
  df.date = df.date.astype('datetime64[ns]')
  df = df.set_index('date')
  df.columns = ['SP', 'Elec_kW', 'Gas']
  train_set = df[:'2018-10-31']
  valid_set = df['2018-11-01':]
  print('Proportion of train_set : {:.2f}%'.format(len(train_set)/len(df)))
  print('Proportion of valid_set : {:.2f}%'.format(len(valid_set)/len(df)))

  def split_sequence(sequence, n_steps):
    x, y = list(), list()
    for i in range(len(sequence)):
      end_ix = i + n_steps
      if end_ix > len(sequence)-1:
          break
      
      seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
      x.append(seq_x)
      y.append(seq_y)
    
    return np.array(x), np.array(y)

  n_steps = 3
  train_x, train_y = split_sequence(train_set.Elec_kW.values,n_steps)
  valid_x, valid_y = split_sequence(valid_set.Elec_kW.values,n_steps)



  train_ds = ElecDataset(train_x.reshape(train_x.shape[0],train_x.shape[1],1),train_y)
  valid_ds = ElecDataset(valid_x.reshape(valid_x.shape[0],valid_x.shape[1],1),valid_y)

  return train_ds, valid_ds

train_ds, test_ds = create_datasets(dataset_path)

dl_train = DataLoader(train_ds, batch_size=2, shuffle=True)
dl_test = DataLoader(test_ds, batch_size=2, shuffle=True)

model = TSMixer(input_size=(3, 1), hidden_size=32, output_size=1, blocks_n=16)
# model = CNN_ForecastNet()

# print(torchsummary.summary(model, (3, 1)))
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

def Train():
    running_loss = .0
    model.train()
    for idx, (inputs,labels) in enumerate(dl_train):
      inputs = inputs.to(dev)
      labels = labels.to(dev)
      optimizer.zero_grad()
      preds = model(inputs.float())
      loss = loss_fn(preds,labels)
      loss.backward()
      optimizer.step()
      running_loss += loss
        
    train_loss = running_loss/len(dl_train)
    
    print(f'train_loss {train_loss}')

def Valid():
    running_loss = .0    
    model.eval()
    with torch.no_grad():
      for idx, (inputs, labels) in enumerate(dl_test):
        inputs = inputs.to(dev)
        labels = labels.to(dev)
        optimizer.zero_grad()
        preds = model(inputs.float())
        loss = loss_fn(preds,labels)
        running_loss += loss
          
      valid_loss = running_loss / len(dl_train)
      print(f'valid_loss {valid_loss}')

def train_loop(dataloader, model, loss_fn, optimizer):
  size = len(dataloader.dataset)
  model.train()
  running_loss = 0
  for batch, (X, y) in enumerate(dataloader):
      # Compute prediction and loss
      pred = model(X)
      loss = loss_fn(pred, y)

      running_loss += loss
      # Backpropagation
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()

      if batch % 100 == 0:
          loss, current = loss.item(), (batch + 1) * len(X)
          # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
  train_loss = running_loss / len(dl_train)
  print(f'Train Loss {train_loss}')

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

epochs = 200
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    # Train()
    train_loop(dl_train, model, loss_fn, optimizer)
    Valid()
    # test_loop(dl_test, model, loss_fn)

print("Done!")
