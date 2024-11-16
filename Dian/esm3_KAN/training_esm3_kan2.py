# Import ESM3 package
from huggingface_hub import login
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization import get_model_tokenizers
import torch
#import KAN package
from fastkan import FastKAN as KAN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
#import regular package
from functools import reduce
import numpy as np
import pandas as pd
import os
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import *

import wandb
# Dataset Class
class PeptideComplexDataset(Dataset):
    def __init__(self, annotations_file, json_dir, peptide_len):
        # df = pd.read_csv(annotations_file, nrows = 4)
        df = pd.read_csv(annotations_file)
        # print(df)
        self.annotations_file = df
        self.json_dir = json_dir
        self.peptide_len = peptide_len

    def __len__(self):
        return len(self.annotations_file)

    def __getitem__(self, idx):
        json_filePath = os.path.join(self.json_dir, self.annotations_file.iloc[idx, 0])
        protein_ID_key = self.annotations_file.iloc[idx, 1]
        combined_chains_key = self.annotations_file.iloc[idx, 2]
        posIdx_binary_key = self.annotations_file.iloc[idx, 2]
        with open(json_filePath, 'r') as file:
            json_data = json.load(file)
            combined_chains = json_data[protein_ID_key]['combined_chains'][combined_chains_key]
            posIdx_binary_str = json_data[protein_ID_key]['posIdx_binary'][posIdx_binary_key]
            posIdx_binary_tensor = torch.tensor(list(map(int, posIdx_binary_str.split())))
            distances = json_data[protein_ID_key]['distance'][combined_chains_key]
            peptide_distances = torch.tensor(distances[:self.peptide_len])
            peptide_distances = torch.clamp(peptide_distances, min=0)
            protein_distances = torch.tensor(distances[self.peptide_len+1:])
            protein_distances = torch.clamp(protein_distances, min=0)

        return combined_chains, posIdx_binary_tensor, peptide_distances, protein_distances

# Model
class esm3_KAN(nn.Module):
    def __init__(self, esm3_model, KAN_model, KAN_model2, batch_size, predLen, numLabelTypes):
        super(esm3_KAN, self).__init__()
        self.esm3_model = esm3_model.train()
        self.KAN_model = KAN_model
        self.KAN_model2 = KAN_model2
        self.flatten = nn.Flatten()
        self.batch_size = batch_size
        self.predLen = predLen
        self.numLabelTypes = numLabelTypes

    def forward(self, x):
        # Pass ESM3 model
        x = self.esm3_model.forward(sequence_tokens=x)
        embeddings = x.embeddings # Shape: [batch_size, sequence_length, hidden_size]
        torch.save(embeddings, '/mnt/c/esm3_KAN2 training/1.pt')
        print(embeddings.shape)
        # Pass KAN model
        embeddings = self.flatten(embeddings)
        print(embeddings.shape)
        out = self.KAN_model(embeddings)
        out = self.KAN_model2(out)
        out = out.view(self.batch_size, self.numLabelTypes, self.predLen)
        return out

def calculate_accuracy(out_argmax, y_batch):
    a_batch_total_ones = torch.sum(y_batch == 1).item()
    a_batch_total_zeros = torch.sum(y_batch == 0).item()
    a_batch_matching_ones = torch.sum((y_batch == 1) & (out_argmax == 1)).item()
    a_batch_matching_zeros = torch.sum((y_batch == 0) & (out_argmax == 0)).item()
    accuracy = (a_batch_matching_ones+a_batch_matching_zeros) / (a_batch_total_ones+a_batch_total_zeros)
    ones_accuracy = a_batch_matching_ones / a_batch_total_ones
    if a_batch_total_zeros==0:
        zeros_accuracy = 0
    else:
        zeros_accuracy = a_batch_matching_zeros / a_batch_total_zeros
    return accuracy, ones_accuracy, zeros_accuracy

def evaluate(model, test_loader, criterion, device, protein_weight, peptide_weight):
    model.eval()
    test_loss = 0
    test_accuracy = 0
    test_ones_accuracy = 0
    test_zeros_accuracy = 0
    
    test_accuracy_peptide = 0
    test_ones_accuracy_peptide = 0
    test_zeros_accuracy_peptide = 0

    test_accuracy_protein = 0
    test_ones_accuracy_protein = 0
    test_zeros_accuracy_protein = 0

    with torch.no_grad():
        for x_batch_test, y_batch_test, peptide_distances, protein_distances in test_loader:
            x_batch_test = torch.stack([torch.tensor(tokenizers.sequence.encode(seq), dtype=torch.int64) for seq in x_batch_test])
            x_batch_test=x_batch_test.to(device)
            out = esm3_KAN(x_batch_test)
            y_batch_test=y_batch_test.to(device)

            out_peptide = out[:,:,:50].to(device)
            out_protein = out[:,:,51:].to(device)
            y_batch_test_peptide = y_batch_train[:, :50].to(device)
            y_batch_test_protein = y_batch_train[:, 51:].to(device)

            loss_peptide= criterion(out_peptide,y_batch_test_peptide)
            loss_protein = criterion(out_protein,y_batch_test_protein)
            test_loss_a_batch = protein_weight*loss_protein + peptide_weight*loss_peptide
            test_loss += test_loss_a_batch.item()
        
            #test_loss += criterion(out, y_batch_test.to(device)).item()
            # whole complex test accuracy
            out_argmax = out.argmax(dim=1)
            test_accuracy_increment, test_ones_accuracy_increment, test_zeros_accuracy_increment = calculate_accuracy(out_argmax, y_batch_test)
            test_accuracy += test_accuracy_increment
            test_ones_accuracy += test_ones_accuracy_increment
            test_zeros_accuracy += test_zeros_accuracy_increment

            # protein test accuracy
            test_accuracy_protein_increment, test_ones_accuracy_protein_increment, test_zeros_accuracy_protein_increment = calculate_accuracy(out_argmax_protein, y_batch_test_protein)
            test_accuracy_protein += test_accuracy_protein_increment
            test_ones_accuracy_protein += test_ones_accuracy_protein_increment
            test_zeros_accuracy_protein += test_zeros_accuracy_protein_increment

            # peptide test accuracy
            out_argmax_peptide = out_argmax[:, :50]
            y_batch_test_peptide = y_batch_test[:, :50]
            test_accuracy_peptide_increment, test_ones_accuracy_peptide_increment, test_zeros_accuracy_peptide_increment = calculate_accuracy(out_argmax_peptide, y_batch_test_peptide)
            test_accuracy_peptide += test_accuracy_peptide_increment
            test_ones_accuracy_peptide += test_ones_accuracy_peptide_increment
            test_zeros_accuracy_peptide += test_zeros_accuracy_peptide_increment

        test_loss /= len(test_loader)
        test_accuracy /= len(test_loader)
        test_zeros_accuracy = test_zeros_accuracy/len(test_loader)
        test_ones_accuracy = test_ones_accuracy/len(test_loader)

        test_accuracy_protein = test_accuracy_protein/len(test_loader)
        test_zeros_accuracy_protein = test_zeros_accuracy_protein/len(test_loader)
        test_ones_accuracy_protein = test_ones_accuracy_protein/len(test_loader)

        test_accuracy_peptide = test_accuracy_peptide/len(test_loader)
        test_zeros_accuracy_peptide = test_zeros_accuracy_peptide/len(test_loader)
        test_ones_accuracy_peptide = test_ones_accuracy_peptide/len(test_loader)

    return test_loss, test_accuracy, test_ones_accuracy, test_zeros_accuracy, test_accuracy_protein, test_ones_accuracy_protein, test_zeros_accuracy_protein, test_accuracy_peptide, test_ones_accuracy_peptide, test_zeros_accuracy_peptide

def calculate_1s0s_ratio(tensor):
    # Ensure the tensor is on the CPU for computation
    tensor = tensor.cpu()
    
    # Flatten the tensor to 1D for easy counting
    flattened_tensor = tensor.flatten()
    
    # Count the number of 1s and 0s
    num_ones = torch.sum(flattened_tensor == 1).item()
    num_zeros = torch.sum(flattened_tensor == 0).item()
    
    # Compute the ratio
    if num_zeros == 0:
        return float('inf')  # Return infinity if there are no zeros to avoid division by zero
    ratio = num_ones / num_zeros
    
    return ratio

# def pytorch_count_params(model):
#     """Count the number of trainable parameters in a PyTorch model."""
#     total_params = sum(
#         reduce(lambda a, b: a * b, x.size()) for x in model.parameters() if x.requires_grad
#     )
#     return total_params
#Main
# Initialization
login(token="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") #hugging face login
annotations_file = '/mnt/f/Dropbox/新建文件夹/Dropbox/2024/Cheng lab/PDB/propedia v2.3/annotation files/annotation_files_esm3.csv'
json_dir = '/mnt/f/Dropbox/新建文件夹/Dropbox/2024/Cheng lab/PDB/propedia v2.3/json_esm3 - Copy'
predicted_dir = '/mnt/f/Dropbox/新建文件夹/Dropbox/2024/Cheng lab/PDB/propedia v2.3/predicted_tensor'
# CUDA device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'
# Load ESM-3 model
tokenizers = get_model_tokenizers()
esm3_model = ESM3_sm_open_v0("cuda").train()
# Set requires_grad for all parameters based on the trainable argument
# for param in esm3_model.parameters():
#     param.requires_grad = True
# print(vars(esm3_model))

esm3_model.to(device)

# # check esm3 trainable
# trainable_params = pytorch_count_params(esm3_model)
# if trainable_params > 0:
#     print(f"The esm3_KAN model is trainable with {trainable_params} trainable parameters.")
# else:
#     print("The esm3_KAN model is not trainable.")

# Load KAN model
d_in=849408 #fixed size from esm3 embeddings
numNeurons=100
numLabelTypes=3
protein_len = 500
peptide_len=50
predLen=protein_len + peptide_len + 1

d_out=predLen*numLabelTypes
KAN_model = KAN([d_in, numNeurons, d_out])
KAN_model.to(device)

KAN_model2 = KAN([d_out, numNeurons, d_out])
KAN_model2.to(device)

# Define loss
# loss part1: peptide-protein loss
weight_scale = 10.0
zeros_weight = weight_scale * 0.2 # ratio between 1s and 0s
ones_weight = weight_scale * 0.8
protein_weight = weight_scale * protein_len/(predLen-1)
peptide_weight = weight_scale * peptide_len/(predLen-1)
weights = torch.tensor([zeros_weight, ones_weight, 0], dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=weights)

# Prepare Dataset
batch_size=2
dataset = PeptideComplexDataset(annotations_file, json_dir, peptide_len)
train_ratio = 0.9
train_size = int(train_ratio * len(dataset))
test_size = len(dataset) - train_size
# Split the dataset
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last = True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last = True)

# Initialize WandB
wandb.init(project='esm3_KAN')

# Training
rates = [1e-3]
num_epoch = 1
num_batches = len(train_loader)
half_epoch = num_batches//2-1

for lr in rates:
    # Create model
    esm3_KAN = esm3_KAN(esm3_model, KAN_model, KAN_model2, batch_size, predLen, numLabelTypes)
    esm3_KAN.to(device)
    # Define optimizer
    optimizer = optim.AdamW(esm3_KAN.parameters(), lr=lr, weight_decay=1e-4)
    # Define learning rate scheduler
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    predicted_tensor_idx=1

    for epoch in range(num_epoch):
      esm3_KAN.train() # put model in training mode
      with tqdm(train_loader) as pbar:
         # iterate over training set
         for i, (x_batch_train, y_batch_train, peptide_distances, protein_distances) in enumerate(pbar):
            # Calculate 0_1 ratio
            ratio = calculate_1s0s_ratio(y_batch_train)
            wandb.log({
                "epoch": epoch,
                "ratio": ratio
            })

            x_batch_train = torch.stack([torch.tensor(tokenizers.sequence.encode(seq), dtype=torch.int64) for seq in x_batch_train])
            x_batch_train = x_batch_train.to(device)
            print("x_batch_train type:", type(x_batch_train))
            print("x_batch_train content:", x_batch_train)
            optimizer.zero_grad()
            out = esm3_KAN(x_batch_train)
            y_batch_train = y_batch_train.to(device)

            # Calculate loss
            # loss part1: prediction loss
            out_peptide = out[:,:,:50]
            out_protein = out[:,:,51:]
            y_batch_train_peptide = y_batch_train[:, :50]
            y_batch_train_protein = y_batch_train[:, 51:]
            loss_peptide_pred = criterion(out_peptide,y_batch_train_peptide)
            loss_protein_pred = criterion(out_protein,y_batch_train_protein)
            loss_prediction = protein_weight*loss_protein_pred + peptide_weight*loss_peptide_pred
            # loss part2: distance loss
            out_argmax = out.argmax(dim=1)
            out_argmax_peptide_interface = out_argmax[:, :peptide_len]
            out_argmax_protein_interface = out_argmax[:, peptide_len+1:]
            # print(out_argmax_peptide[1])
            # print(out_argmax_peptide[1])
            out_argmax_peptide_interface[out_argmax_peptide_interface == 2] = 0
            out_argmax_protein_interface[out_argmax_protein_interface == 2] = 0
            # print(out_argmax_peptide[1])
            # print(out_argmax_peptide[1])
            peptide_distances = peptide_distances.float().to(device)
            protein_distances = protein_distances.float().to(device)
            peptide_distance_loss = torch.sum(torch.dot(out_argmax_peptide_interface.float().flatten(), peptide_distances.flatten()))
            protein_distance_loss = torch.sum(torch.dot(out_argmax_protein_interface.float().flatten(), protein_distances.flatten()))
            peptide_numInterfacePred = torch.sum(out_argmax_peptide_interface)
            protein_numInterfacePred = torch.sum(out_argmax_protein_interface)
            loss_distance = peptide_distance_loss/peptide_numInterfacePred + protein_distance_loss/protein_numInterfacePred
            loss = loss_prediction + loss_distance
            # Compute gradients using back propagation
            loss.backward()
            # Take an optimization 'step'
            optimizer.step()

            out_argmax_peptide = out_argmax[:, :peptide_len]
            out_argmax_protein = out_argmax[:, peptide_len+1:]
            # Calculate accuracy
            train_accuracy, train_ones_accuracy, train_zeros_accuracy = calculate_accuracy(out_argmax, y_batch_train)
            train_accuracy_protein, train_ones_accuracy_protein, train_zeros_accuracy_protein = calculate_accuracy(out_argmax_protein, y_batch_train_protein)
            train_accuracy_peptide, train_ones_accuracy_peptide, train_zeros_accuracy_peptide = calculate_accuracy(out_argmax_protein, y_batch_train_protein)

            # define wandb
            wandb.log({
                "train_loss": loss.item(),
                "train_accuracy":train_accuracy,
                "train_zeros_accuracy": train_zeros_accuracy,
                "train_ones_accuracy": train_ones_accuracy,
                "train_accuracy_protein":train_accuracy_protein,
                "train_zeros_accuracy_protein": train_zeros_accuracy_protein,
                "train_ones_accuracy_protein": train_ones_accuracy_protein,
                "train_accuracy_peptide":train_accuracy_peptide,
                "train_zeros_accuracy_peptide": train_zeros_accuracy_peptide,
                "train_ones_accuracy_peptide": train_ones_accuracy_peptide
            })
            pbar.set_postfix(loss = loss.item(), accuracy_protein = train_accuracy_protein, ones_accuracy_protein = train_ones_accuracy_protein, zeros_accuracy_protein = train_zeros_accuracy_protein, lr = optimizer.param_groups[0]['lr'])

            # Evaluate at half of epoch
            if i == half_epoch or i == num_batches-1:
                predicted_tensor = out_argmax[0]
                predicted_tensor_filePath = os.path.join(predicted_dir, f"predicted_{predicted_tensor_idx}")
                predicted_tensor_idx = predicted_tensor_idx + 1
                predicted_tensor_truth = y_batch_train[0].to(device)
                comparison = predicted_tensor != predicted_tensor_truth
                comparison = comparison.int()
                predicted_tensor_comparison = torch.cat((predicted_tensor, predicted_tensor_truth),dim=0)
                torch.save(predicted_tensor_comparison,predicted_tensor_filePath)

                plt.figure()
                plt.bar(range(len(comparison)), comparison.tolist())
                plt.xlabel('Index')
                plt.ylabel('Value')
                if i == half_epoch:
                    plt.title(f"Test Tensor Difference Half_Epoch {epoch+1}")
                if i == num_batches-1:
                    plt.title(f"Test Tensor Difference End Of Epoch {epoch+1}")
                test_loss, test_accuracy, test_ones_accuracy, test_zeros_accuracy, test_accuracy_protein, test_ones_accuracy_protein, test_zeros_accuracy_protein, test_accuracy_peptide, test_ones_accuracy_peptide, test_zeros_accuracy_peptide = evaluate(esm3_KAN, test_loader, criterion, device, protein_weight, peptide_weight)
                wandb.log({
                    "test_loss": test_loss,
                    "test_accuracy":test_accuracy,
                    "test_ones_accuracy": test_ones_accuracy,
                    "test_zeros_accuracy": test_zeros_accuracy,
                    "test_accuracy_protein":test_accuracy_protein,
                    "test_zeros_accuracy_protein": test_zeros_accuracy_protein,
                    "test_ones_accuracy_protein": test_ones_accuracy_protein,
                    "test_accuracy_peptide":test_accuracy_peptide,
                    "test_zeros_accuracy_peptide": test_zeros_accuracy_peptide,
                    "test_ones_accuracy_peptide": test_ones_accuracy_peptide,
                    f"test_tensor_diff_epoch_{epoch + 1}": wandb.Image(plt)
                })
                print(f"Epoch {epoch + 1}, test Loss: {test_loss}, test accuracy protein: {test_accuracy_protein}, test zero accuracy protein: {test_zeros_accuracy_protein}, test ones accuracy protein: {test_ones_accuracy_protein}")
      # Update learning rate
      scheduler.step()