# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:33:09 2023

@author: Tamara Verbeek
"""
import pandas as pd
import numpy as np
from scipy.stats import percentileofscore
from itertools import islice
import torch
from Data import Data
from Buffer import Buffer
from collections import deque
from SDL import SDL, test_and_update_indices, train_model_MIR, standard_train
import time
import argparse
import matplotlib.pyplot as plt
from PrefixTreeCDDmain.CDD import Window
from sklearn.metrics import accuracy_score, mean_absolute_error, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt
from Utils.LogFile import LogFile
import Setting as setting
from skmultiflow.drift_detection import ADWIN, PageHinkley
from collections import OrderedDict
from PrefixTreeCDDmain.PrefixTreeClass import PrefixTree
from numpy import log as ln
import math
from Transform_data import transform


parser = argparse.ArgumentParser()
parser.add_argument('--result_dir', type=str, default='Results',
    help='directory where we save results and samples')
parser.add_argument('--train_iter', type=int, default=1,
    help='number of training iterations for the classifier')
parser.add_argument('--mem_size', type=int, default=500, help='controls buffer size')
parser.add_argument('--n_classes', type=int, default=-1,
    help='total number of classes. -1 does default amount for the dataset')
parser.add_argument('--buffer_batch_size', type=int, default=10)
parser.add_argument('--trainPerc', type=int, default=0.5)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--learning_rate', type=float, default=0.002)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
args.device = 'cuda:0'

def recCurveFinal(x,a,u_max):
    if x == 0:
        return 1
    return 1 - math.exp(a*(x/u_max-1))

def store_results(file, results):
    with open(file, "w") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")


def store_resultsa(file, results):
    with open(file, "a") as fout:
        for r in results:
            fout.write(",".join([str(r_i) for r_i in r]) + "\n")

def store_timinga(file, timing):
    with open(file, "a") as fout:
        fout.write(str(timing) + "\n")

def store_timing(file, timing):
    with open(file, "w") as fout:
        fout.write(str(timing) + "\n")
        
# Load the XES event log
data = pd.read_csv("Data/BPIC15_next_activity.csv", low_memory=False)
dataName = 'BPIC15'
d = Data(dataName, LogFile(filename="Data/BPIC15_next_activity.csv", delim=",", header=0, rows=None, time_attr="time:timestamp", trace_attr="case:concept:name",
                 activity_attr='concept:name', convert=False))
next_activities = d.logfile.get_column('next_activity')
d.logfile.keep_attributes(['case:concept:name', 'time:timestamp', 'concept:name'])
s = setting.STANDARD

trainPerc = args.trainPerc
s.train_percentage = trainPerc * 100
d.prepare(s)
classes = next_activities.unique()
# Convert the event log to csv
args.n_classes = len(classes)
timeformat = "%Y-%m-%d %H:%M:%S"
numEvents = d.logfile.data.shape[0]

index = 0
buffer = Buffer(args, d.logfile.k)

delay_val = 490
u_min = 200 
u_max = 500 
t_min = 500 
t_max = 750

a = ln(0.2) / ((delay_val /u_max) - 1)
updateArray = range(u_min, u_max)
trainArray = range(t_max, t_min, -1)

transform = transform(data)
input_size = d.logfile.k * len(transform.possible_activities) # Specify your input size
output_size = args.n_classes  # Specify your output size (number of classes)
model = SDL(input_size, output_size) #initialize the model

#Initialization
train_data = d.get_test_batchi(0, int(0.1 *len(d.logfile.data)))
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
model = train_model_MIR(args, buffer, model, train_loader, True)
#model, losses = standard_train(args, model, train_loader)
print("Initialization Complete")

updateInt = 1500
updateWindow = 750
window_size = 10
tree_size = 1000
decay_lambda = 0.25
noise = 1
endEventsDic = dict()
window = Window(initWinSize=window_size)

lastEvents = data.groupby(['case:concept:name']).last()
for _, row in lastEvents.iterrows():
    endEventsDic[_] = [str(row['concept:name']), row['time:timestamp']]

caseList = []  # Complete list of cases seen
Dcase = OrderedDict()  # Dictionary of cases that we're tracking.
# print("You are here")
tree = PrefixTree(pruningSteps=tree_size, noiseFilter=noise,
                  lambdaDecay=decay_lambda)  # Create the prefix tree with the first main node empty
adwin = ADWIN()
ph = PageHinkley()

pruningCounter = 0  # Counter to check if pruning needs to be done
traceCounter = 0  # Counter to create the Heuristics Miner model

eventCounter = 0  # Counter for number of events
currentNode = tree.root  # Start from the root node
testInd = round(numEvents * trainPerc)
prevDriftCounter = 0
drifts = {}
new_drifts = {}
severity = -88
start_time = time.time()
for _, event in islice(data.iterrows(), int(0.1 *len(d.logfile.data)), None):
    if event.get("next_activity") is not None:
        caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree.insertByEvent(caseList, Dcase, currentNode, event, pruningCounter, traceCounter, endEventsDic, window)
        eventCounter += 1
        if window.cddFlag: # If a complete new tree has been created
            if len(window.prefixTreeList) == window.WinSize:
                # Maximum size of window reached, start concept drift detection within the window
                temp_drifts = window.conceptDriftDetection(adwin, ph, eventCounter)
                window.WinSize = min(window.WinSize + 1, window.maxWindowSize)
                for i in temp_drifts.keys():
                    if i not in drifts.keys():
                        new_drifts[i] = temp_drifts[i]
                if len(window.prefixTreeList) == window.WinSize:  # If there was no drift detected within the window
                    window.prefixTreeList = deque(islice(window.prefixTreeList, 1, None))  # Drop the oldest tree
        if len(list(drifts.keys()) + list(new_drifts.keys())) > prevDriftCounter and (_ > round(numEvents * trainPerc)):
            if len(list(drifts.keys())) >= 5:
                #list of drift severities
                drift_sevs = [drifts[i]['treeDist'] for i in list(drifts.keys())]
                new_drift_sevs = [new_drifts[i]['treeDist'] for i in list(new_drifts.keys())]
                severity = percentileofscore(a = drift_sevs, score = sum(new_drift_sevs)/len(new_drift_sevs))/100 #* 5000
                print("Severity is {}".format(severity))
            else:
                severity = 1
                print("Max Severity level")            
        elif (_ - testInd > updateInt) and (_ > round(numEvents * trainPerc)):
            #if last drift greater than predetermined retrain frequency -> assign moderate severity
            if severity == -1 or severity == -88:
                print("Max update interval level")
                severity = 0.75
        elif severity == -88 and (_ > round(numEvents * trainPerc)) and len(list(drifts.keys()))>0:# last drift was < 5000 events ago....then update with the most recent available information
            if _ - sorted([i for i in drifts.keys()])[-1] < 5000:
                severity = 1
        if severity != -1 and severity != -88 and ((_ - testInd > updateInt) or len(list(new_drifts.keys())) > 0):
            severity = min(1, severity)
            print(severity)

            #Update these values and create a list using rec_curve_final
            #severity outputs a score between 0 and 1
            #need to use the update and train array to actually convert to a meaningful update/train value
            print("Update val index value is {}".format(round(recCurveFinal(severity, a, u_max) * len(updateArray)) - 1))
            updateVal = min(updateArray[round(recCurveFinal(severity, a, u_max) * len(updateArray)) - 1], u_max)
            winVal = max(t_min, trainArray[round(recCurveFinal(severity, a, u_max) * len(trainArray)) - 1])
            print("Update Value is {}".format(updateVal))
            print("Window Value is {}".format(winVal))
            updateInt, updateWindow = updateVal, winVal
            #Determine new values for updateInt and updateWindow before running below
            if updateWindow > _ - round(numEvents * trainPerc):
                updateWindow = 0
            result, timing, model = test_and_update_indices(args, model, d, (_ - round(numEvents * trainPerc), max(_ - round(
                                                                             numEvents * trainPerc) - updateWindow, #maximally the total number in the test set to retrain
                                                                             0)),
                                                        testInd - round(numEvents * trainPerc),
                                                        _ - round(numEvents * trainPerc),
                                                        False, transform, input_size, output_size, buffer)
            testInd = _
            end_time = time.time()
            is_written = 1
            severity += 0.1

            if is_written:
                store_resultsa("results/%s_%s_OTF_drift_%s.csv" % (model.name, d.name, 'dynamic'), result)
                store_timinga("results/%s_%s_OTF_drift_%s_time.csv" % (model.name, d.name, 'dynamic'),
                              timing)
            else:
                store_results("results/%s_%s_OTF_drift_%s.csv" % (model.name, d.name, 'dynamic'), result)
                store_timing("results/%s_%s_OTF_drift_%s_time.csv" % (model.name, d.name, 'dynamic'),
                             timing)
        #reset drifts for next run's analysis
        drifts = {**drifts, **new_drifts}
        prevDriftCounter += len(list(new_drifts.keys()))
        new_drifts = {}
        if severity >= 1:
            severity = -1
"""
epochs = range(1, len(avg_acc) + 1)
plt.plot(epochs, avg_acc, 'b', label='acc')
plt.title('Average training accuracy per batch')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Show the plot
plt.savefig('accuracy.png')
""" 
