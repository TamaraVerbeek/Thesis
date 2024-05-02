4# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:55:20 2023

@author: Tamara Verbeek
"""

import argparse
import sys
from Preprocess import Preprocess
import torch
from LSTM import init_model, train_model, VariableBatchSizeDataLoader
import numpy as np
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import csv
from PrefixTreeCDDmain.PrefixTreeClass import PrefixTree
from collections import OrderedDict, deque
from itertools import islice
from PrefixTreeCDDmain.CDD import Window
from PrefixTreeCDDmain.DDScripts import prefixTreeDistances
import datetime
import torch.optim as optim
import time
from torch.utils.data import Subset
import pickle
import copy as cp
import os

def test_old_data(prefixes, target, model, task, metrics, old_task, lengths):

    for oldies in set(old_task):
        X_tensor = torch.tensor(np.array(prefixes[str(oldies)]))
        Y_tensor = torch.tensor(np.array(target[str(oldies)]))
        dataset = TensorDataset(X_tensor, Y_tensor)
        indices = [next((i for i, sublist in enumerate(inp)if any(np.array_equal(sublist, ev) for ev in args.encodedFirstEvents)),-1) for inp in X_tensor]
        buckets = [1 if 0 <= value <= lengths[0] else 2 if lengths[0] < value <= lengths[1] else 3 if lengths[1] < value <= lengths[2] else 0 for value in indices]
        buckets_indices = {bucket: [] for bucket in set(buckets)}
        [buckets_indices[bucket].append(index) for index, bucket in enumerate(buckets, start=0)]
        # Combine all indices of the same bucket into a single sublist
        combined_indices = list(buckets_indices.values())
        batch_sizes = [len(sublist) for sublist in combined_indices]
        bucket_values = []
        for i in range(len(batch_sizes)):
            value = i # Calculate the value based on the index range
            bucket_values.extend([value] * batch_sizes[i])  # Append the value to the new list

        flattened_list = [item for sublist in combined_indices for item in sublist]
        # Sort the dataset based on indices
        sorted_dataset = Subset(dataset, flattened_list)
        
        # create dataloader input for model
        new_data = VariableBatchSizeDataLoader(sorted_dataset, batch_sizes)
        new_data.create_dataloaders()
        
        with torch.no_grad():
            for ind, dataload in enumerate(new_data.dat):
                for inputs, targets in dataload:
                    # Ensure that tensors have the same data type
                    inputs = inputs.to(torch.float32)
                    targets = targets.to(torch.long)
                    outputs = model(inputs, str(task), True, bucket_values[ind])
                    predictions = np.argmax(outputs, axis=1)
                    label = np.argmax(targets.numpy(), axis = 1)
                
        try:
            metrics['new_'+str(task)+'_oldTask_' + str(oldies)].append(accuracy_score(predictions, label))
        except:
            metrics['new_'+str(task)+'_oldTask_' + str(oldies)] = [accuracy_score(predictions, label)]
    return metrics

def main(args, writer, concept_drift, dataset):
    #load the preprocessing class
    pre = Preprocess(args)
    if 'Recurrent' in dataset:
        s = 'recurrent'
    elif 'Random' in dataset:
        s = 'random'
    elif 'Imbalanced' in dataset:
        s = 'imbalanced'
    elif 'BPIC15' in dataset:
        s = '2015'
    elif 'BPIC17' in dataset:
        s = '2017'
    
    print('prefix storage/contextdata_'+ str(s) +'.pkl')
    if os.path.isfile('prefix storage/contextdata_'+ str(s) +'.pkl'):
        dat, listOfEvents = pre.get_data()
        listOfEvents += 1
        data = dat.logfile.data
        with open('prefix storage/contextdata_'+ str(s) +'.pkl', 'rb') as f:
            contextdata = pickle.load(f)
        with open('prefix storage/targets_'+ str(s) +'.pkl', 'rb') as f:
            targets = pickle.load(f)
    else:
        # Get the data
        contextdata, data, targets, listOfEvents = pre.create_k_context()
        listOfEvents += 1
        #with open('prefix storage/contextdata_'+ str(s) +'.pkl', 'wb') as f:
        #    pickle.dump(contextdata, f)
        #with open('prefix storage/targets_'+ str(s) +'.pkl', 'wb') as f:
        #    pickle.dump(targets, f)

    args.nrOfEvents = listOfEvents
    firstEvents = data.groupby(['case']).first()
    firstEvents = list(set(firstEvents['event']))
    firsts = np.array([pre.one_encoder_event.transform(np.array(event).reshape(-1,1)) for event in firstEvents])
    args.encodedFirstEvents = firsts.reshape(len(firsts), listOfEvents)
    
    encoded_targets = targets
    encoded_prefixes = contextdata
    args.input_size = listOfEvents
    args.output_size = listOfEvents
    
    #initialize the window
    w = deque(maxlen=args.window_size)
    init_data = encoded_prefixes[:args.init_size]
    init_targets = encoded_targets[:args.init_size]
    
    #initialize model
    model, criterion, optimizer, lengths = init_model(args, data, pre, w, init_data, init_targets)
    if not args.use_e_prompt and not args.use_g_prompt:
        for name, param in model.named_parameters():
            if 'head' not in name:  
                param.requires_grad = False
    
    if args.opt == 'nadam' and not args.use_g_prompt and not args.use_e_prompt:
        optimizer = optim.NAdam(model.head.parameters(), lr=args.lr, betas=args.opt_betas, eps=args.opt_eps)
    task_optimizer = {}
    task_optimizer['1'] = optimizer
    
    #up until the first concept drift will be initialized as task 1
    task = 1
    check_task = False
    
    #initialize storage for the predictions and actual labels
    all_predictions = deque(maxlen=args.window_size)
    all_labels = deque(maxlen=args.window_size)
    
    #initialize accuracy list
    accuracys = []
    metrics = {}
    
    #initialize the list of tasks we come across in this dataset, where the first task is task 1
    seen_tasks = set()
    tasks = []
    tasks.append(task)
    
    task_buffer = 0
    caseList = []  # Complete list of cases seen
    Dcase = OrderedDict()  # Dictionary of cases that we're tracking.
    tree = PrefixTree(pruningSteps=args.pruningSteps, noiseFilter=args.noiseFilter,lambdaDecay=args.lambdaDecay)
    
    currentNode = tree.root  # Start from the root node
    pruningCounter = 0  # Counter to check if pruning needs to be done
    traceCounter = 0  # Counter to create the Heuristics Miner model
    
    endEventsDic = dict()
    window = Window(initWinSize=args.window_size)
    
    #wrong_inputs = deque(maxlen=50)
    #wrong_targets = deque(maxlen=50)

    lastEvents = data.groupby(['case']).last()
    for _, row in lastEvents.iterrows():
        endEventsDic[_] = [row['event'], row['completeTime']]
    ts = [(1, 0)]
    index = 0
    new=False
    new_task = False
    update = 0
    
    store = False
    storage_pre = {}
    storage_target = {}
    storage_indices = [x - 100 for x in concept_drift]
    update_indices = [x + 100 for x in concept_drift]
    new_tree = None
    seen = False
    
    shows = {'0':0, '1':0, '2':0, '3':0}
    corrects = {'0':0, '1':0, '2':0, '3':0}
    #pres = deque(maxlen = args.batch_size)
    #tars = deque(maxlen = args.batch_size)
    #loop through the data stream
    task_index = 1
    start_index = 0
    up = True
    for ind, d in islice(data.iterrows(), args.init_size, None):
        pres = []
        prefix = encoded_prefixes[ind]
        target = encoded_targets[ind]
        pres.append(prefix)
        X_tensor = torch.tensor(np.array(pres))
        Y_tensor = torch.tensor(np.array([target]))
        dataset = TensorDataset(X_tensor, Y_tensor)
        new_data = DataLoader(dataset, batch_size=1)
#        tars.append(target)
        caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase,
                                                                                                currentNode, d,
                                                                                                pruningCounter,
                                                                                                traceCounter,
                                                                                        endEventsDic, window)

        if ind in storage_indices:
            store = True
            storage_pre[str(task)] = []
            storage_target[str(task)] = []
        if store: 
            storage_pre[str(task)].append(prefix)
            storage_target[str(task)].append(target)
        if ind in concept_drift:
            """
            index = ind%args.window_size
            print(index)
            task = tasks[task_index]
            store = False
            print(f'new task is: {task}')
            up = False
            task_index += 1
            if task in seen_tasks:
                new = True
                seen = True
                if model.e_prompt is not None or model.g_prompt is not None:
                    optimizer = task_optimizer[str(task)]
            else:
                new = True
                if model.e_prompt is not None:
                    model.e_prompt.det_e_prompt_task(str(task))
                if model.e_prompt is not None or model.g_prompt is not None:
                    optimizer = optim.NAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
                    task_optimizer[str(task)] = optimizer
                seen_tasks.add(task)
            """

            store = False
            index = ind%args.window_size
            up = False
            if new_task:
                tree.store_tree(window)
                new_task = False
            if len(tasks) == 1:
                store = False
                task = 2
                max_task = 2
                tasks.append(task)
                ts.append((task, 0))
                tree.store_tree(window)
                if model.e_prompt is not None:
                    model.e_prompt.det_e_prompt_task(str(task))
                if model.e_prompt is not None or model.g_prompt is not None:
                    optimizer = optim.NAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
                    task_optimizer[str(task)] = optimizer
                new = True
            elif len(tasks) == 2:
                metrics = test_old_data(storage_pre, storage_target, model, task, metrics, tasks[:-1], lengths)
                store = False
                tree.store_tree(window)
                win = Window(initWinSize=args.window_size)
                cList = []
                pCounter = 0  # Counter to check if pruning needs to be done
                tCounter = 0  # Counter to create the Heuristics Miner model
                Diccase = OrderedDict()  # Dictionary of cases that we're tracking.
                new_tree = PrefixTree(pruningSteps=args.pruningSteps, noiseFilter=args.noiseFilter,lambdaDecay=args.lambdaDecay)
                cNode = new_tree.root  # Start from the root node
                check_task = True
                task_buffer = 0
            else:
                metrics = test_old_data(storage_pre, storage_target, model, task, metrics, tasks[:-1], lengths)
                store = False
                check_task = True
                win = Window(initWinSize=args.window_size)
                cList = []
                pCounter = 0  # Counter to check if pruning needs to be done
                tCounter = 0  # Counter to create the Heuristics Miner model
                Diccase = OrderedDict()  # Dictionary of cases that we're tracking.
                new_tree = PrefixTree(pruningSteps=args.pruningSteps, noiseFilter=args.noiseFilter,lambdaDecay=args.lambdaDecay)
                cNode = new_tree.root  # Start from the root node
                task_buffer = 0
                
        if check_task:
            task_buffer += 1
            cList, Diccase, cNode, pCounter, tCounter, win = new_tree.insertByEvent(cList, Diccase,cNode, d, pCounter, tCounter, endEventsDic, win)

        if task_buffer == args.buffer_size and check_task:
            W1 = deque([new_tree.get_new_tree()])
            minimum = np.inf
            min_task = 0
            previous_task = tasks[-1]
            store_min = np.inf
            for dr in range(len(set(tasks))):
                if dr != previous_task-1:
                    W0 = deque(islice(window.prefixTreeList, dr, dr+1))
                    Window0, Window1, node0, node1 = window.buildContinMatrix(W0, W1)
                    treeDistance = prefixTreeDistances(Window0, Window1)
                    dist = treeDistance.treeDistanceMetric
                    ts.append((dr+1, dist))
                    print(f'Distance is {dist} to task {dr+1}')
                    if minimum == dist:
                        continue
                    minimum = min(minimum, dist)
                    if dist == minimum:
                        min_task = dr+1
                        store_min = dist
            if store_min < args.boundary:
                task = min_task
                tasks.append(task)
                new = True
                seen = True
                if model.e_prompt is not None or model.g_prompt is not None:
                    optimizer = task_optimizer[str(task)]
                print(f'Data is part of task: {task}')
            else:
                max_task += 1
                task = max_task
                tasks.append(max_task)
                new_task = True
                if model.e_prompt is not None:
                    model.e_prompt.det_e_prompt_task(str(task))
                if model.e_prompt is not None or model.g_prompt is not None:
                    optimizer = optim.NAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
                    task_optimizer[str(max_task)] = optimizer
                new=True
                print('Initializing new task')
            check_task = False
            new_tree.reset()

        #evaluate the output given by the model
        model.eval()

        with torch.no_grad():
            for inputs, targets in new_data:
                index2 = [i for i, sublist in enumerate(inputs[0])if any(np.array_equal(sublist, ev) for ev in args.encodedFirstEvents)]
                if index2 != []:
                    buck = [1 if 0 <= value <= lengths[0] else 2 if lengths[0] < value <= lengths[1] else 3 if lengths[1] < value <= lengths[2]  else 0 for value in index2]
                    buck = buck[0]
                else:
                    buck = 0
                shows[str(buck)] += 1
                # Ensure that tensors have the same data type
                inputs = inputs.to(torch.float32)
                targets = targets.to(torch.long)
                outputs = model(inputs, str(task), check_task, buck)
                predictions = np.argmax(outputs, axis = 1)
                label = np.argmax(targets.numpy(), axis = 1)
                
                if predictions.detach().numpy()[0] == label[0]:
                    corrects[str(buck)] += 1
                #    wrong_inputs.append(inputs[0].detach().numpy())
                #    wrong_targets.append(targets.numpy()[0])
        
        all_predictions.extend(predictions)
        all_labels.extend(label)
    
        # Convert to numpy arrays for scikit-learn functions
        last_preds = np.array(all_predictions)[-250:]
        last_labels = np.array(all_labels)[-250:]

        # Calculate accuracy
        accuracy = accuracy_score(last_labels, last_preds)
        accuracys.append(accuracy)
        
        # Writing the accuracy to a CSV file
        writer.writerow((ind,  accuracy))

        if ind%args.window_size == index and ind >= args.init_size + args.window_size and not check_task and up:
            print(f'Index: {ind}, Accuracy: {accuracy:.4f}')
            prefs = encoded_prefixes[ind-args.window_size:ind]
            targs = encoded_targets[ind-args.window_size:ind]
            model, criterion, optimizer, = train_model(args, model, criterion, optimizer, task, prefs, targs, lengths) 
        elif new:
            print('update after concept drift')
            prefs = encoded_prefixes[ind-args.buffer_size:ind]
            targs = encoded_targets[ind-args.buffer_size:ind]
            model, criterion, optimizer= train_model(args, model, criterion, optimizer, task, prefs, targs, lengths) 
            up = True
            new = False
        elif ind%args.window_size == index and ind > args.init_size+ args.window_size and not check_task:
            print(f'Index: {ind}, Accuracy: {accuracy:.4f}')
            print('not updating')
        w.append(d)
    return accuracys, ts, tasks, metrics
    
if __name__ == '__main__':
    datasets = ['ImbalancedTasks.csv', 'RecurrentTasks.csv', 'RandomTasks.csv', 'BPIC15_recurrent_loop.csv', 'BPIC17.csv'] 
    for d in datasets:
        
        parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
        parser.add_argument('--dataset', type=str, default=d,
            help='dataset')
        args_o = parser.parse_known_args()[0]
        subparser = parser.add_subparsers(dest='subparser_name')
        
        if 'BPIC15' in args_o.dataset:
            from configs.BPIC15_ALL_dualprompt import get_args_parser
            config_parser = subparser.add_parser('BPIC15_ALL_dualprompt', help='BPIC15 all DualPrompt configs')
        elif 'BPIC17' in args_o.dataset:
            from configs.BPIC17 import get_args_parser
            config_parser = subparser.add_parser('BPIC17_dualprompt', help='BPIC17 DualPrompt configs')
        elif args_o.dataset == 'IOR_tasks.csv':
            from configs.IOR import get_args_parser
            config_parser = subparser.add_parser('IOR', help='IOR DualPrompt configs')
        elif args_o.dataset == 'RandomTasks.csv':
            from configs.Woppie import get_args_parser
            config_parser = subparser.add_parser('Woppie', help='Woppie DualPrompt configs')
        elif args_o.dataset == 'InternationalDeclarations.csv':
            from configs.InternationalDeclarations import get_args_parser
            config_parser = subparser.add_parser('InternationalDeclarations', help='InternationalDeclarations DualPrompt configs')
        elif args_o.dataset == 'RecurrentTasks.csv':
            from configs.Looploop import get_args_parser
            config_parser = subparser.add_parser('Looploop_dataset', help='Looploop_dataset DualPrompt configs')
        elif args_o.dataset == 'TravelPermits.csv':
            from configs.TravelPermit import get_args_parser
            config_parser = subparser.add_parser('TravelPermit', help='PermitLog DualPrompt configs')
        elif args_o.dataset == 'DomesticDeclarations.csv':
            from configs.DomesticDeclarations import get_args_parser
            config_parser = subparser.add_parser('DomesticDeclarations', help='DomesticDeclarations DualPrompt configs')
        elif args_o.dataset == 'RequestForPayment.csv':
            from configs.RequestForPayment import get_args_parser
            config_parser = subparser.add_parser('RequestForPayment', help='RequestForPayment DualPrompt configs')
        elif args_o.dataset == 'PrepaidTravelCosts.csv':
            from configs.PrepaidTravelCost import get_args_parser
            config_parser = subparser.add_parser('PrepaidTravelCost', help='PrepaidTravelCost DualPrompt configs')
        elif args_o.dataset == 'ImbalancedTasks.csv':
            from configs.infrequent import get_args_parser
            config_parser = subparser.add_parser('Infrequent', help='Infrequent DualPrompt configs')
    
        
        get_args_parser(config_parser)
        args = config_parser.parse_args()
        
        total_times = []
        t = {}#, 'dualprompt_8':[], 'dualprompt_10':[], 'dualprompt_12':[], 'dualprompt_15':[], 'dualprompt_18':[], 'dualprompt_20':[]}
        accs = {}#, 'dualprompt_8':[], 'dualprompt_10':[], 'dualprompt_12':[], 'dualprompt_15':[], 'dualprompt_18':[], 'dualprompt_20':[]}
        tree_diffs = {}#, 'dualprompt_8':[], 'dualprompt_10':[], 'dualprompt_12':[], 'dualprompt_15':[], 'dualprompt_18':[], 'dualprompt_20':[]}
        dat = args_o.dataset
        args.dataset = args_o.dataset
        #get all concept drifts
        if dat == 'OIR_tasks.csv':
            concept_drift = [5224, 11042, 16247, 22178, 27322, 33187, 38349, 44269, 49451] #OIR
        elif dat == 'IOR_tasks.csv':
            concept_drift = [5224, 10469, 15674, 21002, 26146, 31429, 36591, 41853, 47035] #IOR
        elif dat == 'OIRandIORcombi.csv':
            #concept_drift = [50, 100, 150, 200]
            concept_drift = [5818, 11042, 16860, 22065, 27996] #combi
        elif dat == 'RandomTasks.csv':
            concept_drift = [2724, 5756, 9074, 12025, 15297, 18042, 20680, 23385, 26816, 30068, 33105, 36101, 38786, 41614, 44979, 47623, 50125, 52627, 55129, 57631, 60133, 63493, 66198, 69256, 72060, 75121, 77904]
            taskies = [1, 2, 3, 4, 5, 6, 7, 1, 3, 5, 4, 2, 7, 6, 3, 1, 2, 7, 5, 6 ,4, 5, 1, 4, 7, 2, 6, 3]
        elif dat == 'RecurrentTasks.csv':
            concept_drift = [2725, 5757, 9076, 12073, 15505, 18211, 21273, 24639, 27765, 31186, 33831, 36848, 40232, 42733, 45234, 47735, 50573, 53533, 56333, 59200, 61800, 64522, 67390, 70193, 73136, 75732, 78233, 80734, 83235, 85736, 88237, 90738, 93239, 95740, 98241]
            taskies = [1,2,3,2,3,1,2,3,2,3,1,2,3,2,3,1,2,3,2,3,1,2,3,2,3,1,2,3,2,3,1,2,3,2,3,1]
        elif 'cp_recurring_tasks' in dat:
            concept_drift = [3620, 7676]
        elif 'lp_recurring_tasks' in dat:
            concept_drift = [3620, 8650]
        elif 'OIR_recurring_tasks' in dat:
            concept_drift = [3760, 9000]
        elif 'InternationalDeclarations' in dat:
            concept_drift = [12502]
        elif 'DomesticDeclarations' in dat:
            concept_drift = [9883]
        elif 'PrepaidTravelCosts' in dat:
            concept_drift = [2369]
        elif 'RequestForPayment' in dat:
            concept_drift = [4921]
        elif 'TravelPermits' in dat:
            concept_drift = [13693]
        elif 'BPIC17' in dat:
            concept_drift = []
            taskies = [1]
        elif 'BPIC15_recurrent_loop' in dat:
            concept_drift = [2001, 4002, 6003, 8004, 10005, 12006, 14007, 16008, 18009, 20010, 22011, 24012, 26013, 28014, 30015]#[2724, 5756, 9074, 12025, 15297, 18042, 20680, 23385, 26816, 30068, 33105, 36101, 38786, 41614, 44979, 47623, 50125, 52627, 55129, 57631, 60133, 63493, 66198, 69256, 72060, 75121, 77904]
            taskies = [1,2,3,4,1,2,3,4,1,2,3,4,1,2,3,4]
        elif 'Imbalanced' in dat:
            concept_drift = [2724, 5469, 8501, 11773, 14769, 17474, 19976, 22804, 25528, 28030]
            taskies = [1,2,3,4,3,1,3,2,1,2,1]
        
        start_time = time.time()
    
        start_time = time.time()
        args.use_g_prompt = True
        args.use_e_prompt = True
        if d == 'BPIC17.csv':
            args.window_size = 500
        else:
            args.window_size = 1000
        print('Dual Prompt')
        for j in range(0, 1):
            file_name = 'output/'+args.output_file+str(args.window_size)+'_'+str(d)+'_'+str(j)+'.csv'
            with open(file_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                accuracys, ts, tasks, metrics = main(args, writer, concept_drift, d)
            end_time = time.time()
            total_time = end_time - start_time
            total_time_str = datetime.timedelta(seconds=int(total_time))
            total_times.append(total_time_str)
            t['dualprompt_'+str(d)+'_'+str(j)] = tasks
            tree_diffs['dualprompt_'+str(d)+'_'+str(j)] = ts
            accs['dualprompt_'+str(d)+'_'+str(j)] = sum(accuracys)/len(accuracys)
                    
            print(f'Total running times:{total_times}')
            print(f'Order of tasks:{t}')
            print(f'Average accuracy: {sum(accuracys)/len(accuracys)}')
            print(f'Tasks and tree difference: {tree_diffs}')
            
            # Open the file in write mode
            with open('output/forgetting_prefix_tune_'+str(d)+'_'+str(j)+'.txt', 'w') as file:
                # Convert each element to a string and write to the file
                for key, value in metrics.items():
                    file.write(f"{key}: {value}\n")
                    
        # Open the file in write mode
        with open('output/tasks'+str(d)+'.txt', 'w') as file:
            # Convert each element to a string and write to the file
            for key, value in t.items():
                file.write(f"{key}: {value}\n")
            
        # Open the file in write mode
        with open('output/tree_diffs'+str(d)+'.txt', 'w') as file:
            # Convert each element to a string and write to the file
            for key, value in tree_diffs.items():
                file.write(f"{key}: {value}\n")
                
        # Open the file in write mode
        with open('output/times'+str(d)+'.txt', 'w') as file:
            # Convert each element to a string and write to the file
            for timeee in total_times:
                file.write(f"{timeee}\n")
    """

    print('Dual Prompt')
    e = 10
    g = 5
    args.e_prompt_length = e
    args.g_prompt_length = g
    args.use_e_prompt = True
    args.use_g_prompt = False
    for i in range(0,5):
        file_name = args.output_file+'_dualprompt_'+str(i)+'_wop_only_eprompt_2.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            accuracys, ts, tasks, metrics = main(args, writer, concept_drift, taskies)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = datetime.timedelta(seconds=int(total_time))
        total_times.append(total_time_str)
        t['dualprompt_'+str(i)+'_eprompt'] = tasks
        tree_diffs['dualprompt_'+str(i)+'_eprompt'] = ts
        accs['dualprompt_'+str(i)+'_eprompt'] = sum(accuracys)/len(accuracys)
                
        print(f'Total running times:{total_times}')
        print(f'Order of tasks:{t}')
        print(f'Average accuracy: {sum(accuracys)/len(accuracys)}')
        print(f'Tasks and tree difference: {tree_diffs}')
        
    args.use_e_prompt = False
    args.use_g_prompt = True
    for i in range(0,5):
        file_name = args.output_file+'_dualprompt_'+str(i)+'_wop_only_gprompt_2.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            accuracys, ts, tasks, metrics = main(args, writer, concept_drift, taskies)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = datetime.timedelta(seconds=int(total_time))
        total_times.append(total_time_str)
        t['dualprompt_'+str(i)+'_gprompt'] = tasks
        tree_diffs['dualprompt_'+str(i)+'_gprompt'] = ts
        accs['dualprompt_'+str(i)+'_gprompt'] = sum(accuracys)/len(accuracys)
                
        print(f'Total running times:{total_times}')
        print(f'Order of tasks:{t}')
        print(f'Average accuracy: {sum(accuracys)/len(accuracys)}')
        print(f'Tasks and tree difference: {tree_diffs}')
    args.use_e_prompt = True
    args.use_g_prompt = True
    for i in range(0,5):
        file_name = args.output_file+'_dualprompt_'+str(i)+'_wop_dualprompt_2.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            accuracys, ts, tasks, metrics = main(args, writer, concept_drift, taskies)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = datetime.timedelta(seconds=int(total_time))
        total_times.append(total_time_str)
        t['dualprompt_'+str(i)+'_dualprompt'] = tasks
        tree_diffs['dualprompt_'+str(i)+'_dualprompt'] = ts
        accs['dualprompt_'+str(i)+'_dualprompt'] = sum(accuracys)/len(accuracys)
                
        print(f'Total running times:{total_times}')
        print(f'Order of tasks:{t}')
        print(f'Average accuracy: {sum(accuracys)/len(accuracys)}')
        print(f'Tasks and tree difference: {tree_diffs}')

    args.use_e_prompt = False
    args.use_g_prompt = False
    for i in range(0,5):
        file_name = args.output_file+'_dualprompt_'+str(i)+'_wop_noprompt_2.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            accuracys, ts, tasks, metrics = main(args, writer, concept_drift, taskies)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = datetime.timedelta(seconds=int(total_time))
        total_times.append(total_time_str)
        t['dualprompt_'+str(i)+'_noprompt'] = tasks
        tree_diffs['dualprompt_'+str(i)+'noprompt'] = ts
        accs['dualprompt_'+str(i)+'_noprompt'] = sum(accuracys)/len(accuracys)
                
        print(f'Total running times:{total_times}')
        print(f'Order of tasks:{t}')
        print(f'Average accuracy: {sum(accuracys)/len(accuracys)}')
        print(f'Tasks and tree difference: {tree_diffs}')

    start_time = time.time()
    print('Dual Prompt')
    args.prefix = False
    file_name = args.output_file+'_dualprompt_2015loop_'+str(args.prefix)+'.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        accuracys, ts, tasks, metrics = main(args, writer, concept_drift)
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = datetime.timedelta(seconds=int(total_time))
    total_times.append(total_time_str)
    t['dualprompt_false'].append(tasks)
    tree_diffs['dualprompt_false'].append(ts)
    accs['dualprompt_false'].append(sum(accuracys)/len(accuracys))
            
    print(f'Total running times:{total_times}')
    print(f'Order of tasks:{t}')
    print(f'Average accuracy: {sum(accuracys)/len(accuracys)}')
    print(f'Tasks and tree difference: {tree_diffs}')
    """
    """
    start_time = time.time()
    args.use_g_prompt = False
    args.use_e_prompt = True
    for i in range(0,5):
        print('E-Prompt')
        file_name = args.output_file+'_eprompt_loop_'+str(i)+'.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            accuracys, ts, tasks, metrics = main(args, writer, concept_drift)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = datetime.timedelta(seconds=int(total_time))
        total_times.append(total_time_str)
        t['eprompt'].append(tasks)
        tree_diffs['eprompt'].append(ts)
        accs['eprompt'].append(sum(accuracys)/len(accuracys))
                
        print(f'Total running times:{total_times}')
        print(f'Order of tasks:{t}')
        print(f'Average accuracy: {sum(accuracys)/len(accuracys)}')
        print(f'Tasks and tree difference: {tree_diffs}')

    start_time = time.time()
    args.use_g_prompt = True
    args.use_e_prompt = False
    for i in range(0,5):
        print('G-Prompt')
        file_name = args.output_file+'_gprompt_loop_'+str(i)+'.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            accuracys, ts, tasks, metrics = main(args, writer, concept_drift)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = datetime.timedelta(seconds=int(total_time))
        total_times.append(total_time_str)
        t['gprompt'].append(tasks)
        tree_diffs['gprompt'].append(ts)
        accs['gprompt'].append(sum(accuracys)/len(accuracys))
                
        print(f'Total running times:{total_times}')
        print(f'Order of tasks:{t}')
        print(f'Average accuracy: {sum(accuracys)/len(accuracys)}')
        print(f'Tasks and tree difference: {tree_diffs}')

    start_time = time.time()
    args.use_g_prompt = False
    args.use_e_prompt = False
    for i in range(0,5):
        print('G-Prompt')
        file_name = args.output_file+'_noprompt_loop_'+str(i)+'.csv'
        with open(file_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            accuracys, ts, tasks, metrics = main(args, writer, concept_drift)
        end_time = time.time()
        total_time = end_time - start_time
        total_time_str = datetime.timedelta(seconds=int(total_time))
        total_times.append(total_time_str)
        t['noprompt'].append(tasks)
        tree_diffs['noprompt'].append(ts)
        accs['noprompt'].append(sum(accuracys)/len(accuracys))
                
        print(f'Total running times:{total_times}')
        print(f'Order of tasks:{t}')
        print(f'Average accuracy: {sum(accuracys)/len(accuracys)}')
        print(f'Tasks and tree difference: {tree_diffs}')
    
    start_time = time.time()
    args.use_g_prompt = True
    args.use_e_prompt = False
    args.g_prompt_length = 12
    file_name = args.output_file+'_gprompt_length12.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        accuracys, ts, tasks, metrics = main(args, writer, concept_drift)
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = datetime.timedelta(seconds=int(total_time))
    total_times.append(total_time_str)
    t['result4'] = tasks
    tree_diffs['result4'] = ts
    accs['result4'] = sum(accuracys)/len(accuracys)

        
    print(f'Total running times:{total_times}')
    print(f'Order of tasks:{t}')
    print(f'Tasks and tree difference: {tree_diffs}')

    start_time = time.time()
    args.use_g_prompt = True
    args.use_e_prompt = False
    args.g_prompt_length = 15
    file_name = args.output_file+'_gprompt_length15.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        accuracys, ts, tasks, metrics = main(args, writer, concept_drift)
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = datetime.timedelta(seconds=int(total_time))
    total_times.append(total_time_str)
    t['result5'] = tasks
    tree_diffs['result5'] = ts
    accs['result5'] = sum(accuracys)/len(accuracys)

    print(f'Total running times:{total_times}')
    print(f'Order of tasks:{t}')
    print(f'Tasks and tree difference: {tree_diffs}')
    """
        
    print(accs)
    sys.exit(0)
