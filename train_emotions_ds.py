from tsmixer_model import TSMixer
from local_datasets import SnoozeDataset
import pandas as pd
import datetime
import sys
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model_cpkt = 'distilbert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_cpkt)
model = AutoModel.from_pretrained(model_cpkt).to(dev)

def create_datasets():
  emotions_ds = load_dataset('emotion')
  emotions_encoded = emotions_ds.map(lambda _e: tokenizer(_e['text'], padding='max_length', truncation=True), batched=True, batch_size=None)
  emotions_encoded.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])
  emotions_hidden = emotions_encoded.map(
     lambda _e: { "embeddings": model(
        **{ 
           k: v.to(dev) for k,v in _e.items() if k in tokenizer.model_input_names
        }
      ).last_hidden_state.detach() },
  batched=True, batch_size=32)

  return train_ds, test_ds

train_ds, test_ds = create_datasets()

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
