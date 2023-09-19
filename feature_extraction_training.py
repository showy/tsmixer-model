from transformers import AutoModel
from transformers import AutoTokenizer
from datasets import load_dataset
from datasets import load_from_disk
from umap import UMAP
from sklearn.preprocessing import MinMaxScaler
import torch
import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, precision_score, recall_score
import matplotlib.pyplot as plt

emotions = load_dataset("emotion")
model_ckpt = "distilbert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_ckpt)


def tokenize(batch):
  return tokenizer(batch["text"], padding=True, truncation=True)

def extract_hidden_states(batch):
  inputs = { k: v.to(device) for k,v in batch.items() if k in tokenizer.model_input_names }
  with torch.no_grad():
    last_hidden_state = model(**inputs).last_hidden_state

  return { "hidden_state": last_hidden_state[:, 0].cpu().numpy() }

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
emotions_hidden = emotions_encoded.map(extract_hidden_states, batched=True)

# Saving emotions_hidden dataset to disk because the mapping takes time
emotions_ds_filename = 'emotions_hidden'
emotions_hidden.save_to_disk(emotions_ds_filename)


X_train = np.array(emotions_hidden['train']['hidden_state'])
X_valid = np.array(emotions_hidden['validation']['hidden_state'])
y_train = np.array(emotions_hidden['train']['label'])
y_valid = np.array(emotions_hidden['validation']['label'])

X_scaled = MinMaxScaler().fit_transform(X_train)

mapper = UMAP(n_components=2, metric='cosine').fit(X_scaled)

df_emb = pd.DataFrame(mapper.embedding_, columns=['X', 'Y'])
df_emb.loc[:, 'label'] = y_train

fig, axes = plt.subplots(2, 3, figsize=(7, 5))
axes = axes.flatten()

#cmaps = [ 'Reds', 'Purples', 'Orange', 'Pinks', 'Blues', 'Yellows' ]
cmaps = ["Greys", "Blues", "Oranges", "Reds", "Purples", "Greens"]
names = list(sorted(emotions_hidden['train'].features['label'].names))

for idx, (cmap, label_str) in enumerate(zip(cmaps, names)):
  label_idx = emotions_hidden['test'].features['label'].names.index(label_str)
  df_emb_sub = df_emb.query(f"label == {label_idx}")
  axes[idx].hexbin( df_emb_sub.X, df_emb_sub.Y, cmap=cmap, label=label_str, gridsize=20 )
  axes[idx].set_xticks([])
  axes[idx].set_yticks([])
  axes[idx].set_title(label_str)


plt.tight_layout()
plt.show()

lr_clf = LogisticRegression(max_iter=3000)
lr_clf.fit(X_train, y_train)

recall_score(y_valid, lr_clf.predict(X_valid), average='weighted') ,precision_score(y_valid, lr_clf.predict(X_valid), average='weighted')

def confusion_matrix_dialog(preds, y_true, labels):
  cm = confusion_matrix(y_true, preds, normalize='true')
  fig, axes = plt.subplots(figsize=(6, 6))
  cmd = ConfusionMatrixDisplay(cm, display_labels=labels)
  cmd.plot(cmap='Purples', ax=axes, colorbar=False)
  plt.title('Confusion Matrix')
  plt.show()

confusion_matrix_dialog(lr_clf.predict(X_valid), y_valid, labels)
