from transformers import BertTokenizer, BertModel
import torch
import re
import pandas as pd
import os
import json
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim

class PeptideComplexDataset(Dataset):
    def __init__(self, annotations_file, json_dir,tokenizer):
        self.annotations_file=pd.read_csv(annotations_file)
        self.json_dir=json_dir
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, idx):
        json_filePath=os.path.join(self.json_dir, self.annotations_file.iloc[idx, 0])
        protein_ID_key=self.annotations_file.iloc[idx, 1]
        combined_chains_key=self.annotations_file.iloc[idx, 2]
        posIdx_binary_key=self.annotations_file.iloc[idx, 3]
        with open(json_filePath, 'r') as file:
            json_data = json.load(file)
            combined_chains=json_data[protein_ID_key][combined_chains][combined_chains_key]
            combined_chains_tokenized= self.tokenizer.batch_encode_plus(combined_chains, add_special_tokens=True, padding=True, return_tensors='pt')
            posIdx_binary=json_data[protein_ID_key][posIdx_binary][posIdx_binary_key]
        return combined_chains_tokenized, posIdx_binary

class DumbNet(nn.Module):
    def __init__(self, pretrained_model, nh, d_out, epsilon=1e-5, leaky_relu_slope=0.01, drop_rate=0.5):
        super(DumbNet, self).__init__()
        self.epsilon = epsilon
        self.leaky_relu = nn.LeakyReLU(negative_slope=leaky_relu_slope)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(drop_rate)

        # Load pretrained BERT model
        self.pretrained = pretrained_model
        
        # Update input dimension to match the output of the pretrained model
        d_in = self.pretrained.config.hidden_size

        self.Dense1 = nn.Linear(d_in, nh)
        self.Dense2 = nn.Linear(nh, nh)
        self.Dense3 = nn.Linear(nh, nh)
        self.Dense4 = nn.Linear(nh, nh)
        self.Dense5 = nn.Linear(nh, nh)
        self.Dense6 = nn.Linear(nh, nh)
        self.Dense7 = nn.Linear(nh, nh)
        self.Dense8 = nn.Linear(nh, d_out)

    def manual_batch_norm(self, x):
        mean = x.mean(dim=0, keepdim=True)
        var = x.var(dim=0, keepdim=True, unbiased=False)
        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        return x_normalized

    def forward(self, x):
        # Pass through pretrained BERT model
        with torch.no_grad():
            outputs = self.pretrained(x)
            x = outputs.last_hidden_state[:, 0, :]  # Take the embeddings of the [CLS] token

        x = self.dropout(self.leaky_relu(self.manual_batch_norm(self.Dense1(x))))
        x = self.dropout(self.leaky_relu(self.manual_batch_norm(self.Dense2(x))))
        x = self.dropout(self.leaky_relu(self.manual_batch_norm(self.Dense3(x))))
        x = self.dropout(self.leaky_relu(self.manual_batch_norm(self.Dense4(x))))
        x = self.dropout(self.leaky_relu(self.manual_batch_norm(self.Dense5(x))))
        x = self.dropout(self.leaky_relu(self.manual_batch_norm(self.Dense6(x))))
        x = self.dropout(self.leaky_relu(self.manual_batch_norm(self.Dense7(x))))
        out = self.leaky_relu(self.Dense8(x))
        return out

#main
#Initialization
annotations_file = r'A:\Research\Cheng lab\PDB\propedia v2.3\annotation files\annotation_files.csv'
json_dir = r'A:\Research\Cheng lab\PDB\propedia v2.3\json'
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
pretrained_model = BertModel.from_pretrained("Rostlab/prot_bert_bfd")

train_dataset = PeptideComplexDataset(annotations_file, json_dir, tokenizer)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
pretrained_model.to(device)

rates = [1e-2,1e-3,1e-4]
loss_hist = []
val_acc_hist = []

# TODO
num_epoch = 100
with open('model_training_details.txt', 'w') as f:
  for lr in rates:
    model = DumbNet(pretrained_model, d_in=d_in,nh=nh,d_out=d_out) 
    opt = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    a_tr_loss = []
    a_tr_accuracy = np.zeros([num_epoch])
    a_ts_loss = np.zeros([num_epoch])
    a_ts_accuracy = np.zeros([num_epoch])
    for epoch in range(num_epoch):
      model.train() # put model in training mode
      correct = 0 # initialize error counter
      total = 0 # initialize total counter
      # iterate over training set
      for train_iter, data in enumerate(train_loader):
        x_batch,y_batch = data
        y_batch = y_batch.type(torch.long)
        out = model(x_batch)
        # Compute Loss
        loss = criterion(out,y_batch)
        a_tr_loss.append(loss.item())
        # Compute gradients using back propagation
        opt.zero_grad()
        loss.backward()
        # Take an optimization 'step'
        opt.step()
        # Do hard classification: index of largest score
        _, predicted = torch.max(out.data, 1)
        # Compute number of decision errors
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()
      a_tr_accuracy[epoch] = 100*correct/total
      model.eval() # put model in evaluation mode
      correct = 0 # initialize error counter
      total = 0 # initialize total counter
      batch_loss_ts = []
      with torch.no_grad():
        for data in test_loader:
          images, labels = data
          labels = labels.type(torch.long)
          outputs = model(images)
          batch_loss_ts.append(criterion(outputs,labels).item())
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
      a_ts_loss[epoch] = np.mean(batch_loss_ts)
      a_ts_accuracy[epoch] = 100*correct/total
      print('Epoch: {0:2d}   Train Loss: {1:.3f}   '.format(epoch+1, a_tr_loss[epoch])
            +'Train Acc: {0:.2f}    Test Loss: {1:.3f}   '.format(a_tr_accuracy[epoch], a_ts_loss[epoch])
            +'Test Acc: {0:.2f}'.format(a_ts_accuracy[epoch]))
    model_savepath = f'model_lr_{lr:.0e}.pth'
    f.write(f"Learning Rate: {lr}, Epoch: {num_epoch}, Train Loss: {a_tr_loss[-1]:.3f}, "
                f"Train Acc: {a_tr_accuracy[-1]:.2f}, Test Loss: {a_ts_loss[-1]:.3f}, "
                f"Test Acc: {a_ts_accuracy[-1]:.2f}\n")
    # To save a PyTorch model, we first pass an input through the model, 
    # and then save the "trace". 
    # For this purpose, we can use any input. 
    # We will create a random input with the proper dimension.
    x = torch.randn(d_in) # random input
    x = x[None,:] # add singleton batch index
    with torch.no_grad():
        traced_cell = torch.jit.trace(model, (x))

    # Now we save the trace
    torch.jit.save(traced_cell, model_savepath)

    loss_hist.append(a_tr_loss)
    val_acc_hist.append(a_ts_accuracy)