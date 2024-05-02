# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 12:55:20 2023

@author: Tamara Verbeek
"""

import argparse
import sys
from Preprocess import Preprocess
import torch
from LSTM import init_model, train_model
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

def test_old_data(prefixes, target, model, task, metrics, old_task):
    for oldies in set(old_task):
        X_tensor = torch.tensor(np.array(prefixes[str(oldies)]))
        Y_tensor = torch.tensor(target[str(oldies)])
        dataset = TensorDataset(X_tensor, Y_tensor)
        new_data = DataLoader(dataset, batch_size=len(prefixes[str(oldies)]))
        
        with torch.no_grad():
            for inputs, targets in new_data:
                
                # Ensure that tensors have the same data type
                inputs = inputs.to(torch.float32)
                targets = targets.to(torch.long)
                outputs = model(inputs, str(task), True)
                predictions = np.argmax(outputs)
                label = np.argmax(targets.numpy())
                
        metrics[str(task)+'_newTask_' + str(oldies)] = accuracy_score(predictions, label)
    return metrics

def main(args, writer, concept_drift):
    #load the preprocessing class
    pre = Preprocess(args)
    # Get the data
    data, listOfEvents = pre.get_data()
    args.input_size = listOfEvents
    args.output_size = listOfEvents
    
    #initialize the window
    w = deque(maxlen=args.window_size)
    
    da = data.logfile.data #.head(200)
    encoded_prefixes = []
    encoded_targets = []
    for ind, d in da.iterrows():
        prefix, target, _ = pre.get_encoding_new(args, d, ind, data.logfile.data)
        encoded_prefixes.append(prefix)
        encoded_targets.append(target)
    
    init_data = encoded_prefixes[:args.init_size]
    init_targets = encoded_targets[:args.init_size]
    #initialize model
    model, criterion, optimizer = init_model(args, data, pre, w, init_data, init_targets)
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
    
    wrong_inputs = deque(maxlen=50)
    wrong_targets = deque(maxlen=50)

    lastEvents = data.logfile.data.groupby(['case']).last()
    for _, row in lastEvents.iterrows():
        endEventsDic[_] = [str(row['event']), row['completeTime']]
    ts = [(1, 0)]
    index = 0
    new=False
    new_task = False
    
    store = False
    storage_pre = {}
    storage_target = {}
    storage_indices = [x - 100 for x in concept_drift]
    
    cList = []
    pCounter = 0  # Counter to check if pruning needs to be done
    tCounter = 0  # Counter to create the Heuristics Miner model
    Diccase = OrderedDict()  # Dictionary of cases that we're tracking.
    new_tree = None
    cNode = None  # Start from the root node
    #pres = deque(maxlen = args.batch_size)
    #tars = deque(maxlen = args.batch_size)
    #loop through the data stream
    for ind, d in islice(da.iterrows(), args.init_size, None):
        pres = []
        prefix = encoded_prefixes[ind]
        target = encoded_targets[ind]
        pres.append(prefix)
        X_tensor = torch.tensor(np.array(pres))
        Y_tensor = torch.tensor([target])
        dataset = TensorDataset(X_tensor, Y_tensor)
        new_data = DataLoader(dataset, batch_size=1)
#        tars.append(target)
        
        caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase,
                                                                                                currentNode, d,
                                                                                                pruningCounter,
                                                                                                traceCounter,
                                                                                                endEventsDic, window)
        if check_task:
            task_buffer += 1
            cList, Diccase, cNode, pCounter, tCounter, window = new_tree.insertByEvent(cList, Diccase,cNode, d, pCounter, tCounter, endEventsDic, window)

        
        if ind in storage_indices:
            store = True
            storage_pre[str(task)] = []
            storage_target[str(task)] = []
        if store: 
            storage_pre[str(task)].append(prefix)
            storage_target[str(task)].append(target)
        if ind in concept_drift:            
            wrong_inputs = deque(maxlen=50)
            wrong_targets = deque(maxlen=50)
            index = ind%250
            store_window = window
            if new_task:
                tree.store_tree(store_window)
                new_task = False
            if len(tasks) == 1:
                store = False
                task = 2
                max_task = 2
                tasks.append(task)
                ts.append((task, 0))
                tree.store_tree(store_window)
                if model.e_prompt is not None:
                    weights, key = model.e_prompt.det_e_prompt_task(str(task))
                optimizer = optim.NAdam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08)
                task_optimizer[str(task)] = optimizer
                new = True
            elif len(tasks) == 2:
                #metrics = test_old_data(storage_pre, storage_target, model, task, metrics, tasks[:-1])
                #print(metrics)
                store = False
                tree.store_tree(store_window)
                cList = []
                pCounter = 0  # Counter to check if pruning needs to be done
                tCounter = 0  # Counter to create the Heuristics Miner model
                Diccase = OrderedDict()  # Dictionary of cases that we're tracking.
                new_tree = PrefixTree(pruningSteps=args.pruningSteps, noiseFilter=args.noiseFilter,lambdaDecay=args.lambdaDecay)
                cNode = new_tree.root  # Start from the root node
                check_task = True
                task_buffer = 0
            else:
                #metrics = test_old_data(storage_pre, storage_target, model, task, metrics, tasks[:-1])
                #print(metrics)
                store = False
                check_task = True
                cList = []
                pCounter = 0  # Counter to check if pruning needs to be done
                tCounter = 0  # Counter to create the Heuristics Miner model
                Diccase = OrderedDict()  # Dictionary of cases that we're tracking.
                new_tree = PrefixTree(pruningSteps=args.pruningSteps, noiseFilter=args.noiseFilter,lambdaDecay=args.lambdaDecay)
                cNode = new_tree.root  # Start from the root node
                task_buffer = 0
                
        if task_buffer == args.buffer_size and check_task:
            W1 = deque([new_tree.get_new_tree()])
            minimum = np.inf
            min_task = 0
            previous_task = tasks[-1]
            store_min = np.inf
            for dr in range(len(set(tasks))):
                if dr != previous_task-1:
                    W0 = deque(islice(window.prefixTreeList, dr, dr+1))
                    Window0, Window1 = window.buildContinMatrix(W0, W1)
                    treeDistance = prefixTreeDistances(Window0, Window1)
                    dist = treeDistance.treeDistanceMetric
                    ts.append((dr+1, dist))
                    print(f'Distance is {dist} to task {dr+1}')
                    if dist == minimum:
                        continue
                    minimum = min(minimum, dist)
                    if dist == minimum:
                        min_task = dr+1
                        store_min = dist
            if store_min < 1:
                task = min_task
                tasks.append(task)
                optimizer = task_optimizer[str(task)]
                new=True
                print(f'Data is part of task: {task}')
            else:
                max_task += 1
                task = max_task
                tasks.append(max_task)
                new_task = True
                if model.e_prompt is not None:
                    weights, key =  model.e_prompt.det_e_prompt_task(str(task))
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
                
                # Ensure that tensors have the same data type
                inputs = inputs.to(torch.float32)
                targets = targets.to(torch.long)
                outputs = model(inputs, str(task), check_task)
                predictions = np.argmax(outputs, axis = 1)
                label = np.argmax(targets.numpy(), axis = 1)
                
                #if predictions != label:
                #    wrong_inputs.append(inputs[0].detach().numpy())
                #    wrong_targets.append(targets.numpy()[0])
        
        all_predictions.extend(predictions)
        all_labels.extend(label)
    
        # Convert to numpy arrays for scikit-learn functions
        last_preds = np.array(all_predictions)[-args.window_size:]
        last_labels = np.array(all_labels)[-args.window_size:]

        # Calculate accuracy
        accuracy = accuracy_score(last_labels, last_preds)
        
        # Writing the accuracy to a CSV file
        writer.writerow((ind,  accuracy))
            
        if ind%args.window_size == 0 and ind > args.init_size and index < 150:
            print(f'Index: {ind}, Accuracy: {accuracy:.4f}')
            wrongs = zip(wrong_inputs, wrong_targets)
            prefs = encoded_prefixes[ind-args.window_size+index:ind]
            targs = encoded_targets[ind-args.window_size+index:ind]
            model, criterion, optimizer, new = train_model(args, model, pre, w, criterion, optimizer, task, index, wrongs, new, prefs, targs) 
            index = 0
        elif ind%args.window_size == 0 and ind > args.init_size:
            print(f'Index: {ind}, Accuracy: {accuracy:.4f}')
            index = 0
        w.append(d)
    return accuracys, ts, tasks
    
if __name__ == '__main__':
 
    parser = argparse.ArgumentParser('DualPrompt training and evaluation configs')
    parser.add_argument('--dataset', type=str, default='IOR_tasks',
        help='dataset')
    args_o = parser.parse_known_args()[0]
    subparser = parser.add_subparsers(dest='subparser_name')
    
    if args_o.dataset == 'BPIC15_ALL_dualprompt':
        from configs.BPIC15_ALL_dualprompt import get_args_parser
        config_parser = subparser.add_parser('BPIC15_ALL_dualprompt', help='BPIC15 all DualPrompt configs')
    elif args_o.dataset == 'IOR_tasks':
        from configs.IOR import get_args_parser
        config_parser = subparser.add_parser('IOR', help='IOR DualPrompt configs')
    
    get_args_parser(config_parser)
    args = config_parser.parse_args()
    
    total_times = []
    t = {}
    tree_diffs = {}
    dat = 'looploop_dataset.csv'
    args.dataset = dat

    #get all concept drifts
    if dat == 'OIR_tasks.csv':
        concept_drift = [5224, 11042, 16247, 22178, 27322, 33187, 38349, 44269, 49451] #OIR
    elif dat == 'IOR_tasks.csv':
        concept_drift = [5224, 10469, 15674, 21002, 26146, 31429, 36591, 41853, 47035] #IOR
    elif dat == 'OIRandIORcombi.csv':
        #concept_drift = [50, 100, 150, 200]
        concept_drift = [5818, 11042, 16860, 22065, 27996] #combi
    elif dat == 'dataset.csv':
        concept_drift = [5224, 10756, 16575, 22027, 27233, 32766, 38012, 43151, 48357, 54289, 59786, 65324, 71190, 76687, 81873, 87018] #combi
    elif dat == 'loop_dataset.csv':
        concept_drift = [5224, 10756, 16574, 22071, 28003, 33209, 38771, 44637, 50263, 56184, 61329, 65522, 71406, 76903, 82835]
    elif dat == 'woppie.csv':
        concept_drift = [2724, 5756, 9075, 12027, 15300, 18046, 20685, 23391, 26823, 30076, 33114, 36111, 38797, 41626, 44992, 47637, 50138, 52639, 55140, 57641, 60142, 63503, 66209, 69268, 72073, 75135, 77919]
    elif dat == 'looploop_dataset.csv':
        concept_drift = [2725, 5757, 9076, 12073, 15505, 18211, 21273, 24639, 27765, 31186, 33831, 36848, 40232, 42733, 45234, 47735, 50573, 53533, 56333, 59200, 61800, 64522, 67390, 70193, 73136, 75732, 78233, 80734, 83235, 85736, 88237, 90738, 93239, 95740, 98241]
    """
    start_time = time.time()
    args.use_g_prompt = False
    args.use_e_prompt = True
    file_name = args.output_file+'_loopie_woopie_e.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        accuracys, ts, tasks = main(args, writer, concept_drift)
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = datetime.timedelta(seconds=int(total_time))
    total_times.append(total_time_str)
    t['result'] = tasks
    tree_diffs['result'] = ts
        
    print(f'Total running times:{total_times}')
    print(f'Order of tasks:{t}')
    print(f'Tasks and tree difference: {tree_diffs}')
 
    start_time = time.time()
    args.use_g_prompt = True
    args.use_e_prompt = False
    file_name = args.output_file+'_loopie_woopie_g.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        accuracys, ts, tasks = main(args, writer, concept_drift)
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = datetime.timedelta(seconds=int(total_time))
    total_times.append(total_time_str)
    t['result'] = tasks
    tree_diffs['result'] = ts
        
    print(f'Total running times:{total_times}')
    print(f'Order of tasks:{t}')
    print(f'Tasks and tree difference: {tree_diffs}')
    """

    start_time = time.time()
    args.use_g_prompt = True
    args.use_e_prompt = True
    file_name = args.output_file+'_randomness_all.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        accuracys, ts, tasks = main(args, writer, concept_drift)
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = datetime.timedelta(seconds=int(total_time))
    total_times.append(total_time_str)
    t['result'] = tasks
    tree_diffs['result'] = ts
        
    print(f'Total running times:{total_times}')
    print(f'Order of tasks:{t}')
    print(f'Tasks and tree difference: {tree_diffs}')
    
    start_time = time.time()
    args.use_g_prompt = True
    args.use_e_prompt = False
    file_name = args.output_file+'_randomness_g.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        accuracys, ts, tasks = main(args, writer, concept_drift)
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = datetime.timedelta(seconds=int(total_time))
    total_times.append(total_time_str)
    t['result'] = tasks
    tree_diffs['result'] = ts
        
    print(f'Total running times:{total_times}')
    print(f'Order of tasks:{t}')
    print(f'Tasks and tree difference: {tree_diffs}')
    
    start_time = time.time()
    args.use_g_prompt = False
    args.use_e_prompt = True
    file_name = args.output_file+'_randomness_e.csv'
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        accuracys, ts, tasks = main(args, writer, concept_drift)
    end_time = time.time()
    total_time = end_time - start_time
    total_time_str = datetime.timedelta(seconds=int(total_time))
    total_times.append(total_time_str)
    t['result'] = tasks
    tree_diffs['result'] = ts
        
    print(f'Total running times:{total_times}')
    print(f'Order of tasks:{t}')
    print(f'Tasks and tree difference: {tree_diffs}')
    """
    # Open the file in write mode
    with open('tasks.txt', 'w') as file:
        # Convert each element to a string and write to the file
        for key, value in t.items():
            file.write(f"{key}: {value}\n")
        
    # Open the file in write mode
    with open('tree_diffs.txt', 'w') as file:
        # Convert each element to a string and write to the file
        for key, value in tree_diffs.items():
            file.write(f"{key}: {value}\n")
            
    # Open the file in write mode
    with open('times.txt', 'w') as file:
        # Convert each element to a string and write to the file
        for timeee in total_times:
            file.write(f"{timeee}\n")
    
    sys.exit(0)
