import os
import json
import h5py
import numpy as np
import torch
from torch import nn
from transformers import BertTokenizerFast, BertModel
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# 加载 BERT 分词器和模型
checkpoint = 'unikei/bert-base-proteins'
tokenizer = BertTokenizerFast.from_pretrained(checkpoint)
bert_model = BertModel.from_pretrained(checkpoint)

# 自定义数据集类
class PeptideComplexDataset(Dataset):
    def __init__(self, sequences, distance_maps, labels):
        self.sequences = sequences
        self.distance_maps = distance_maps
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        distance_map = self.distance_maps[idx]
        label = self.labels[idx]
        return sequence, distance_map, label

# 数据准备
def prepare_data(json_file, h5_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    h5f = h5py.File(h5_file, 'r')

    sequences = []
    distance_maps = []
    labels = []

    for pdb_id, pdb_data in data.items():
        peptide_chain_id = list(pdb_data["Input"].keys())[0]
        protein_chain_id = list(pdb_data["Input"].keys())[1]

        # 获取肽段和蛋白质序列
        peptide_sequence = pdb_data["Input"][peptide_chain_id]
        protein_sequence = pdb_data["Input"][protein_chain_id]

        # 去掉序列中的残基ID，只保留氨基酸序列
        peptide_seq = ''.join([res.split(':')[0] for res in peptide_sequence.split()])
        protein_seq = ''.join([res.split(':')[0] for res in protein_sequence.split()])

        # 获取距离矩阵
        if pdb_id in h5f:
            distance_map = np.array(h5f[pdb_id]['distance_map'])
        else:
            continue

        sequences.append((peptide_seq, protein_seq))
        distance_maps.append(distance_map)
        labels.append(1)  # 这里假设所有样本的标签为1，可以根据需要修改

    h5f.close()
    return sequences, distance_maps, labels

# 数据处理
def tokenize_sequences(sequences, tokenizer):
    tokenized_peptides = []
    tokenized_proteins = []
    for peptide_seq, protein_seq in sequences:
        peptide_tokens = tokenizer(peptide_seq, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        protein_tokens = tokenizer(protein_seq, return_tensors='pt', padding='max_length', max_length=512, truncation=True)
        tokenized_peptides.append(peptide_tokens)
        tokenized_proteins.append(protein_tokens)
    return tokenized_peptides, tokenized_proteins

# 准备数据
json_file = "output_interface.json"
h5_file = "output.h5"
sequences, distance_maps, labels = prepare_data(json_file, h5_file)
tokenized_peptides, tokenized_proteins = tokenize_sequences(sequences, tokenizer)

# 创建数据集和数据加载器
dataset = PeptideComplexDataset(tokenized_peptides, distance_maps, labels)
train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# 定义模型
class ScoreModel(nn.Module):
    def __init__(self, bert_model):
        super(ScoreModel, self).__init__()
        self.bert = bert_model
        self.fc = nn.Linear(768 + 768 + 512*512, 1)  # 假设最后一层大小为768，加上距离矩阵展平后的大小

    def forward(self, peptide_tokens, protein_tokens, distance_map):
        peptide_output = self.bert(**peptide_tokens).last_hidden_state[:, 0, :]  # 取 [CLS] token 的输出
        protein_output = self.bert(**protein_tokens).last_hidden_state[:, 0, :]  # 取 [CLS] token 的输出
        distance_map_flat = distance_map.view(distance_map.size(0), -1)  # 展平距离矩阵
        combined_input = torch.cat((peptide_output, protein_output, distance_map_flat), dim=1)
        score = self.fc(combined_input)
        return score

# 创建模型实例
model = ScoreModel(bert_model)
criterion = nn.MSELoss()  # 假设使用均方误差损失，可以根据需要更改
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# 训练模型
def train_model(model, train_loader, criterion, optimizer, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for peptide_tokens, protein_tokens, distance_map, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(peptide_tokens, protein_tokens, distance_map)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader)}")

# 评估模型
def evaluate_model(model, test_loader):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for peptide_tokens, protein_tokens, distance_map, labels in test_loader:
            outputs = model(peptide_tokens, protein_tokens, distance_map)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    print(f"Test Loss: {total_loss/len(test_loader)}")

# 运行训练和评估
train_model(model, train_loader, criterion, optimizer)
evaluate_model(model, test_loader)
