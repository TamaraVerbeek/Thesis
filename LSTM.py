# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 10:22:16 2023

@author: Tamara Verbeek
"""
import torch.nn as nn
import torch
import numpy as np
import pandas as pd
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from EPrompt import E_Prompt
from sklearn.metrics import accuracy_score
from itertools import islice
from collections import Counter
import math
from G_Prompt import G_Prompt
from itertools import groupby, accumulate
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_name(i))
    print(torch.cuda.get_device_capability(i))

class VariableBatchSizeDataLoader:
    def __init__(self, dataset, batch_sizes):
        self.dataset = dataset
        self.batch_sizes = batch_sizes
        self.num_batches = len(self.batch_sizes)
        self.dat = []
        self.bucket = []

    def create_dataloaders(self):
        start_idx = 0
        for batch_size in self.batch_sizes:
            end_idx = start_idx + batch_size
            self.dat.append(DataLoader(TensorDataset(self.dataset[start_idx:end_idx][0], self.dataset[start_idx:end_idx][1]), batch_size=batch_size, shuffle=False))
            start_idx = end_idx

    def __len__(self):
        return self.num_batches


# Define MSA model
class LSTMModel(nn.Module):
    def __init__(self, args, input_size, output_size, use_g_prompt=False, g_prompt_layer_idx=None,
                 use_e_prompt=False, e_prompt_layer_idx=None, g_prompt_length=None, prompt_init='uniform', prompt_length=None,
                 qkv_bias = False, prompt_key_init='uniform', num_heads=1):
        super(LSTMModel, self).__init__()
        self.embedding_layers = nn.ModuleList()
        for k in range(args.prefix_length):
            embedding_layer = nn.Embedding(args.nrOfEvents, args.nrOfEvents)
            self.embedding_layers.append(embedding_layer)
        
        self.prefix = args.prefix
        self.num_heads = args.num_heads
        head_dim = input_size // self.num_heads
        self.scale = head_dim ** -0.05
        self.g_prompt_length = args.g_prompt_length
        self.e_prompt_length = args.e_prompt_length 
        self.prompt_prefix_size= args.prompt_prefix_size
        
        self.qkv = nn.Linear(input_size, input_size * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(args.attn_drop)
        
        if use_g_prompt:
            self.g_prompt = G_Prompt(self.num_heads, input_size, ['k', 'v'], prompt_init, g_prompt_length, self.prefix)
            self.g_prompt.init_g_prompt()
        else:
            self.g_prompt = None
        
        #if not self.prefix:
        #    input_size = input_size + args.g_prompt_length
        self.qkv2 = nn.Linear(input_size, input_size * 3, bias=qkv_bias)
        self.attn_drop2 = nn.Dropout(args.attn_drop)
        
        if use_e_prompt:
            self.e_prompt = E_Prompt(self.num_heads, input_size, prompt_key_init, ['k', 'v'], prompt_init, self.e_prompt_length, self.prompt_prefix_size, self.prefix)
            self.e_prompt.init_e_prompt()
        else:
            self.e_prompt = None
            
        #if not self.prefix:
        #    input_size = input_size + args.size + args.prompt_prefix_size
        self.proj2 = nn.Linear(input_size, args.hidden_size)
        self.proj_drop2 = nn.Dropout(args.proj_drop)

        self.norm = nn.LayerNorm(args.hidden_size)
        self.head = nn.Linear(args.hidden_size, output_size)

        
        
            

    def forward(self, x, task, check_task, bucket):
        #x = x.to(torch.long)
        #embedded = [emb_layer(x[:,i]) for i, emb_layer in enumerate(self.embedding_layers)]
        #x = torch.stack(embedded, dim=1)
        #x = x.reshape(x.shape[0], x.shape[1], x.shape[3])
        #appending the g-prompt
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.prefix and self.g_prompt is not None:
            key_prefix, value_prefix = self.g_prompt.get_g_prompt()
            k = torch.cat([key_prefix.expand(len(x), -1, -1, -1), k], dim=2)
            v = torch.cat([value_prefix.expand(len(x), -1, -1, -1), v], dim=2)
        elif self.g_prompt is not None:
            query_prefix, key_prefix, value_prefix = self.g_prompt.get_g_prompt()
            k = torch.cat([key_prefix.expand(len(x), -1, -1, -1), k], dim=2)
            v = torch.cat([value_prefix.expand(len(x), -1, -1, -1), v], dim=2)
            q = torch.cat([query_prefix.expand(len(x), -1, -1, -1), q], dim = 2)
            N += self.g_prompt_length
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        #appending the e-prompt
        B, N, C = x.shape
        qkv2 = self.qkv2(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv2.unbind(0)
        if self.prefix and self.e_prompt is not None and not check_task:
            key_prefix, value_prefix = self.e_prompt.get_e_prompt(task, bucket)
            k = torch.cat([key_prefix.expand(len(x), -1, -1, -1), k], dim=2)
            v = torch.cat([value_prefix.expand(len(x), -1, -1, -1), v], dim=2)
        elif not self.prefix and self.e_prompt is not None: # and not check_task:
            query_prefix, key_prefix, value_prefix = self.e_prompt.get_e_prompt(task, bucket)
            k = torch.cat([key_prefix.expand(len(x), -1, -1, -1), k], dim=2)
            v = torch.cat([value_prefix.expand(len(x), -1, -1, -1), v], dim=2)
            q = torch.cat([query_prefix.expand(len(x), -1, -1, -1), q], dim = 2)
            N += self.e_prompt_length + self.prompt_prefix_size
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop2(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        x = self.proj2(x)
        x = self.proj_drop2(x)
        x = self.norm(x)
        out = self.head(x[:, -1, :])
        
        return out

def split_value(value, num_sub_values):
    # Determine the integer division and remainder
    quotient = value // num_sub_values
    remainder = value % num_sub_values

    # Create a list with the quotient repeated num_sub_values times
    sub_values = [quotient] * num_sub_values

    # Distribute the remainder evenly among the sub-values
    for i in range(remainder):
        sub_values[i] += 1

    return sub_values

def split_values(values, target_value, keys):
    result = []
    values2 = []

    for ind, value in enumerate(values):
        # Determine the number of sub-values needed
        num_sub_values = value // target_value
        if num_sub_values == 0:
            sub_values = [value]
            values2.extend([keys[ind]])
        else:
            remainder = value % target_value
    
            # Distribute the remainder evenly among the sub-values
            sub_values = [target_value] * num_sub_values
            sort = split_value(remainder, num_sub_values)
            for i,s in enumerate(sort):
                sub_values[i] += s
            values2.extend([keys[ind]]*num_sub_values)


        result.extend(sub_values)

    return result, values2

def split_into_pieces_greedy(values, num_pieces):
    total_values = sum(values)
    average_values_per_piece = total_values // num_pieces
    pieces = []
    start_index = 1
    for _ in range(num_pieces - 1):
        piece_size = None
        for i, x in enumerate(values, start=start_index):
            if abs(average_values_per_piece - sum(values[start_index:i + 1])) <= abs(average_values_per_piece - sum(values[start_index:i + 2])):
                piece_size = i        
                pieces.append(piece_size)
                start_index = piece_size + 1
                break
    pieces.append(len(values) - 1)  # Add the last index to cover all remaining values
    return pieces

def freeze_weights(args, model):
    if not args.use_e_prompt and args.use_g_prompt:
        for name, param in model.named_parameters():
            if 'e_prompt' in name or 'qkv2' in name:  
                param.requires_grad = False
                print(param)
    elif not args.use_g_prompt and args.use_e_prompt:
        for name, param in model.named_parameters():
            if 'g_prompt' in name or 'qkv' in name: 
                param.requires_grad = False
                print(param)
    return model
    
def get_batch_sizes(args, input, lengths, init):
    indices = [next((i for i, sublist in enumerate(inp)if any(np.array_equal(sublist, ev) for ev in args.encodedFirstEvents)),-1) for inp in input]
    groups = pd.Series(indices).value_counts().sort_index()
    if init:
        lengths = split_into_pieces_greedy(groups, 4)
    buckets = [1 if 0 <= value <= lengths[0] else 2 if lengths[0] < value <= lengths[1] else 3 if lengths[1] < value <= lengths[2] else 0 for value in indices]
    # Create a dictionary to map buckets to indices
    buckets_indices = {bucket: [] for bucket in set(buckets)}
    [buckets_indices[bucket].append(index) for index, bucket in enumerate(buckets, start=0)]
    # Combine all indices of the same bucket into a single sublist
    combined_indices = list(buckets_indices.values())
    batch_sizes = [len(sublist) for sublist in combined_indices]
    batch_sizes, bucket_values = split_values(batch_sizes, args.batch_size, list(buckets_indices.keys()))
    return batch_sizes, combined_indices, buckets_indices, bucket_values, lengths

def init_model(args, data, pre, window, prefixes, targets):
    
    print('Initializing model...')
    X_tensor = torch.tensor(np.array(prefixes))
    Y_tensor = torch.tensor(np.array(targets))
    dataset = TensorDataset(X_tensor, Y_tensor)
    #train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    # get input and output sizes
    input_size = args.input_size  
    output_size = args.output_size
    batch_sizes, indices, buckets_indices, bucket_values, lengths = get_batch_sizes(args, X_tensor, lengths=None, init=True)
    flattened_list = [item for sublist in indices for item in sublist]
    # Sort the dataset based on indices
    sorted_dataset = Subset(dataset, flattened_list)
    
    # create dataloader input for model
    new_data = VariableBatchSizeDataLoader(sorted_dataset, batch_sizes)
    new_data.create_dataloaders()
    
    # Set a random seed (you can use any integer)
    random_seed = 42
    random.seed(random_seed)
    
    # Shuffle both lists in the same way
    random.shuffle(new_data.dat)
    random.seed(random_seed)  # Reset the seed to ensure the same shuffle order
    random.shuffle(bucket_values)
    #new_data = DataLoader(dataset, batch_size=args.init_batch_size)
    # create standard model
    model = LSTMModel(args, input_size, output_size, use_g_prompt=args.use_g_prompt,
                      g_prompt_layer_idx=args.g_prompt_layer_idx, use_e_prompt=args.use_e_prompt, 
                      e_prompt_layer_idx=args.e_prompt_layer_idx, g_prompt_length=args.g_prompt_length, 
                      prompt_init='uniform', prompt_length=args.length, num_heads=args.num_heads)
    model.to(device)

    # Define the loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    if args.opt == 'nadam':
        optimizer = optim.NAdam(model.parameters(), lr=args.lr, betas=args.opt_betas, eps=args.opt_eps)
    elif args.opt == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=args.opt_betas, eps=args.opt_eps, weight_decay=args.weight_decay)
    
    model.train()  # Set the model to training mode

    # Training loop
    for epoch in range(args.init_epochs):
        for ind, dataload in enumerate(new_data.dat):
            for inputs, targets in dataload:
    
                # Ensure that tensors have the same data type
                inputs = inputs.to(torch.float32)
                #inputs = inputs.to(device)
    
                targets = targets.to(torch.long)
                targets = (targets.nonzero()[:, 1]).view(-1)
                # Forward pass
                outputs = model(inputs, str(1), False, bucket_values[ind])
                #outputs = outputs.cpu()
    
                l2_regularization = sum((param ** 2).sum() for param in model.parameters())
                loss = criterion(outputs, targets) #+ 0.001 * l2_regularization
    
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    model = freeze_weights(args, model)
    return model, criterion, optimizer, lengths


def train_model(args, model, criterion, optimizer, task, prefixes, targets, lengths):
    X_tensor = torch.tensor(np.array(prefixes))
    Y_tensor = torch.tensor(np.array(targets))
    dataset = TensorDataset(X_tensor, Y_tensor)
    #train_loader = DataLoader(dataset, batch_size=32, shuffle=False)

    batch_sizes, indices, buckets_indices, bucket_values, lengths = get_batch_sizes(args, X_tensor, lengths, init=False)
    
    flattened_list = [item for sublist in indices for item in sublist]
    # Sort the dataset based on indices
    sorted_dataset = Subset(dataset, flattened_list)
    
    # create dataloader input for model
    new_data = VariableBatchSizeDataLoader(sorted_dataset, batch_sizes)
    new_data.create_dataloaders()

    # Set a random seed (you can use any integer)
    random_seed = 42
    random.seed(random_seed)
    
    # Shuffle both lists in the same way
    random.shuffle(new_data.dat)
    random.seed(random_seed)  # Reset the seed to ensure the same shuffle order
    random.shuffle(bucket_values)

    model.train()  # Set the model to training mode

    # Training loop
    for epoch in range(args.epochs):
        for ind, dataload in enumerate(new_data.dat):
            for inputs, targets in dataload:
                
                # Ensure that tensors have the same data type
                inputs = inputs.to(torch.float32)
                targets = targets.to(torch.long)
                targets = np.argmax(targets, axis = 1).view(-1)
                
                # Send the input data to the GPU
                #inputs = inputs.to(device)
                #targets = targets.view(-1)
                # Forward pass
                outputs = model(inputs, str(task), False, bucket_values[ind])
                #outputs = outputs.cpu()
                l2_regularization = sum((param ** 2).sum() for param in model.parameters())
    
                loss = criterion(outputs, targets) #+ 0.001 * l2_regularization #+ l1_lambda * sum(torch.norm(param, p=2) for param in w)
                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    
                # Print training statistics
                #print(f'Epoch [{epoch + 1}/{epochs}], bucket: {bucket_values[ind]}, Loss: {loss.item():.4f}')

    return model, criterion, optimizer
