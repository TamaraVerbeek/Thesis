# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 10:47:53 2023

@author: tamar
"""
import torch
import torch.nn as nn
import copy
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score, mean_squared_error
from torch.utils.data import DataLoader, Dataset
import time
import pandas as pd
from itertools import chain

entropy_fn = lambda x : torch.sum(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1), dim=-1)

class CustomDataset(Dataset):
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.length = len(input_data)  # Ensure both input and output have the same length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_sample = self.input_data[idx]
        output_sample = self.output_data[idx]
        return input_sample, output_sample

class SDL(nn.Module):
    def __init__(self, input_size, output_size):
        super(SDL, self).__init__()
        self.name = "SDL"
        
        # Define a single dense (fully connected) layer
        self.fc = nn.Linear(input_size, output_size)
        
        # Define a dropout layer with a dropout rate of 0.2
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # Pass the input through the dense layer
        x = self.fc(x)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Apply softmax activation to the output
#        x = torch.softmax(x, dim=1)
        
        return x
    
def test_and_update_indices(args, model, data, evWindow, prevTestInd, currentTestInd, reset, transform, input_size, output_size, buffer):
    try:
        model = copy.deepcopy(model)
    except:
        import tensorflow as tf
        model.save("tmp_model")
        model = tf.keras.models.load_model('tmp_model')
    timings = []
    predict_batch = data.get_test_batchi(prevTestInd, currentTestInd)
    all_input = predict_batch.contextdata  
    filtered_dataframes = []
    next_act = []
    # Iterate over each DataFrame in x
    for df_name, df in all_input.iterrows():
        # Filter rows in the DataFrame where the index (row names) contain the pattern
        filtered_df = df[df.index.str.contains('concept:name_Prev')]
        filtered_df = filtered_df[~filtered_df.index.str.contains('case')]
        
        # Store the filtered DataFrame in the dictionary with the same name
        filtered_dataframes.append(filtered_df.values)
        next_act.append(df['concept:name'])
        
    inp = pd.DataFrame(filtered_dataframes)
    out = next_act
    test_loader = transform.create_train_set(inp, out, len(inp), False)
    results = test(model, test_loader)
    start_time = time.time()
    
    
    train_data = data.get_test_batchi(evWindow[1], evWindow[0])
    all_input = train_data.contextdata    
    filtered_dataframes = []
    next_act = []
    # Iterate over each DataFrame in x
    for df_name, df in all_input.iterrows():
        # Filter rows in the DataFrame where the index (row names) contain the pattern
        filtered_df = df[df.index.str.contains('concept:name_Prev')]
        filtered_df = filtered_df[~filtered_df.index.str.contains('case')]
        
        # Store the filtered DataFrame in the dictionary with the same name
        filtered_dataframes.append(filtered_df.values)
        next_act.append(df['concept:name'])
        
    inp = pd.DataFrame(filtered_dataframes)
    out = next_act
    train_loader = transform.create_train_set(inp, out, args.batch_size, True)
    #if reset:
    #model, losses = standard_train(args, model, train_loader)
    #else:
    model = train_model_MIR(args, buffer, model, train_loader, init = False)
    timings = time.time() - start_time
    return results, timings, model

def standard_train(args, model, train_loader, criterion, optimizer):
    model.train()
    for epoch in range(args.num_epochs):
        losses = []
        for batch_idx, (data, target) in enumerate(train_loader):
            accs = []
            target = target.long()
                
            # Zero the parameter gradients
            optimizer.zero_grad()
    
            # Forward pass
            outputs = model(data)
            ys = torch.argmax(outputs, dim=1)
    
            # Calculate the loss and accuracy
            loss = criterion(outputs, target.view(-1))
            sdlbatch_accuracy = accuracy_score(ys.detach().numpy(), target.view(-1).detach().numpy())
    
            # Backpropagation and optimization
            loss.backward()
            optimizer.step()
    
            accs.append(sdlbatch_accuracy)
            losses.append(loss.item())

    return model, losses

def test(model, test_loader):
    for batch_idx, (data, target) in enumerate(test_loader):
        predictions = model(data)
        predict_vals = np.argmax(predictions.detach().numpy(), axis=1)
        predict_probs = predictions[np.arange(predictions.shape[0]), predict_vals]
        expected_vals = target
        expected_probs = predictions[np.arange(predictions.shape[0]), expected_vals]
        result = zip(expected_vals, predict_vals, predict_probs, expected_probs)
    return result

def get_grad_vector(pp, grad_dims):
    """
     gather the gradients in one vector
    """
    grads = torch.Tensor(sum(grad_dims))
    #if args.cuda: grads = grads.cuda()

    grads.fill_(0.0)
    cnt = 0
    for param in pp():
        if param.grad is not None:
            beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
            en = sum(grad_dims[:cnt + 1])
            grads[beg: en].copy_(param.grad.data.view(-1))
        cnt += 1
    return grads

def overwrite_grad(pp, new_grad, grad_dims):
    """
        This is used to overwrite the gradients with a new gradient
        vector, whenever violations occur.
        pp: parameters
        newgrad: corrected gradient
        grad_dims: list storing number of parameters at each layer
    """
    cnt = 0
    for param in pp():
        param.grad=torch.zeros_like(param.data)
        beg = 0 if cnt == 0 else sum(grad_dims[:cnt])
        en = sum(grad_dims[:cnt + 1])
        this_grad = new_grad[beg: en].contiguous().view(
            param.data.size())
        param.grad.data.copy_(this_grad)
        cnt += 1
        
def get_future_step_parameters(this_net, grad_vector, grad_dims, lr=1):
    """
    computes \theta-\delta\theta
    :param this_net:
    :param grad_vector:
    :return:
    """
    new_net=copy.deepcopy(this_net)
    overwrite_grad(new_net.parameters,grad_vector,grad_dims)
    with torch.no_grad():
        for param in new_net.parameters():
            if param.grad is not None:
                param.data=param.data - lr*param.grad.data
    return new_net

def retrieve_replay_update(args, buffer, criterion, model, opt, input_x, input_y, rehearse, loader = None):
    # Forward pass
    outputs = model(input_x)
    input_y = input_y.long()
    ys = torch.argmax(outputs, dim=1)
    loss_a = F.cross_entropy(outputs, input_y.view(-1), reduction='none')
    loss = (loss_a).sum() / loss_a.size(0)
    
    sdlbatch_accuracy = accuracy_score(ys.detach().numpy(), input_y.view(-1).detach().numpy())
    opt.zero_grad()
    loss.backward(retain_graph=True)
    if not rehearse:
        opt.step()
        return model, 0
    
    grad_dims = []
    
    bx, by = buffer.sample(100)
    for param in model.parameters():
        grad_dims.append(param.data.numel())
    grad_vector = get_grad_vector(model.parameters, grad_dims)
    model_temp = get_future_step_parameters(model, grad_vector,grad_dims, lr=args.learning_rate)

    with torch.no_grad():
        logits_track_pre = model(bx)
        logits_track_post = model_temp(bx)

        pre_loss = F.cross_entropy(logits_track_pre, by, reduction='none')
        post_loss = F.cross_entropy(logits_track_post, by, reduction='none')
        scores = post_loss - pre_loss
        EN_logits = entropy_fn(logits_track_pre)
                
        all_logits = scores
        big_ind = all_logits.sort(descending=True)[1][:args.buffer_batch_size]

        #idx = subsample[big_ind]

    mem_x, mem_y = bx[big_ind], by[big_ind] #, buffer.logits[idx]
    logits_buffer = model(mem_x)
    ys = torch.argmax(logits_buffer, dim=1)
    sdlbatch_accuracy = accuracy_score(ys.detach().numpy(), mem_y.detach().numpy())
    F.cross_entropy(logits_buffer, mem_y).backward(retain_graph=True)
    return model, sdlbatch_accuracy

def train_model_MIR(args, buffer, model, train_loader, init):
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.NAdam(model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-08)
    # Training loop
    if init: 
        model, losses = standard_train(args, model, train_loader, criterion, optimizer)
    for batch_idx, (data, target) in enumerate(train_loader):
        if init:
            buffer.add_reservoir(data, target, None)
        else:
            for epoch in range(args.num_epochs):
                model, sdlbatch_accuracy = retrieve_replay_update(args, buffer, criterion, model, optimizer, data, target, train_loader)  
            buffer.add_reservoir(data, target, None)

    # Save the trained model if needed
    torch.save(model.state_dict(), 'trained_model.pth')
    return model