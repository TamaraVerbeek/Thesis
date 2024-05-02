# -*- coding: utf-8 -*-
"""
Created on Fri Feb  9 16:32:54 2024

@author: tamar
"""
from PrefixTreeCDDmain.PrefixTreeClass import PrefixTree
from collections import OrderedDict, deque
from PrefixTreeCDDmain.CDD import Window
from Data import Data
from Utils.LogFile import LogFile
import numpy as np
"""
d = Data('woppie',
         LogFile(filename='local_datasets/woppie.csv', delim=",", header=0, rows=None, time_attr="completeTime", trace_attr="case",
                 activity_attr='event', convert=False))
data1 = d.logfile.data.iloc[36105: 36600]
data2 = d.logfile.data.iloc[38000: 38500]

caseList = []  # Complete list of cases seen
Dcase = OrderedDict()  # Dictionary of cases that we're tracking.
tree3 = PrefixTree(pruningSteps=1000, noiseFilter=1,lambdaDecay=0.25)

currentNode = tree3.root  # Start from the root node
pruningCounter = 0  # Counter to check if pruning needs to be done
traceCounter = 0  # Counter to create the Heuristics Miner model
endEventsDic = dict()
window = Window(initWinSize=350)

lastEvents = d.logfile.data.groupby(['case']).last()
for _, row in lastEvents.iterrows():
    endEventsDic[_] = [str(row['event']), row['completeTime']]

for da in data1.iterrows():
    if da[1]['event'] == "Added_activity":
        print('stop')
    caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree3.insertByEvent(caseList, Dcase,
                                                                                            currentNode, da[1],
                                                                                            pruningCounter,
                                                                                            traceCounter,
                                                                                            endEventsDic, window)


caseList = []  # Complete list of cases seen
Dcase = OrderedDict()  # Dictionary of cases that we're tracking.
tree2 = PrefixTree(pruningSteps=1000, noiseFilter=1,lambdaDecay=0.25)

currentNode = tree2.root  # Start from the root node
pruningCounter = 0  # Counter to check if pruning needs to be done
traceCounter = 0  # Counter to create the Heuristics Miner model
window = Window(initWinSize=350)
for da in data2.iterrows():
    caseList, Dcase, currentNode, pruningCounter, traceCounter, window = tree2.insertByEvent(caseList, Dcase,
                                                                                            currentNode, da[1],
                                                                                            pruningCounter,
                                                                                            traceCounter,
                                                                                            endEventsDic, window)
"""
import matplotlib.pyplot as plt

# Define the data from the LaTeX table
data = [
    [None, None, None, None, None, None, None, None],
    [0.78, None, None, None, None, None, None, None],
    [0.51, 0.48, None, None, None, None, None, None],
    [0.79, 0.69, 0.61, None, None, None, None, None],
    [0.78, 0.71, 0.36, 0.72, None, None, None, None],
    [0.70, 0.69, 0.40, 0.62, 0.59, None, None, None],
    [0.68, 0.64, 0.57, 0.63, 0.61, 0.62, None, None],
    [0.77, 0.80, 0.69, 0.71, 0.71, 0.71, 0.43, None],
    [0.56, 0.72, 0.76, 0.63, 0.69, 0.67, 0.45, None],
    [0.42, 0.66, 0.67, 0.76, 0.81, 0.73, 0.56, None],
    [0.49, 0.73, 0.29, 0.83, 0.84, 0.72, 0.64, None],
    [0.49, 0.87, 0.53, 0.83, 0.72, 0.72, 0.53, None],
    [0.52, 0.86, 0.27, 0.82, 0.68, 0.66, 0.79, None],
    [0.45, 0.72, 0.54, 0.70, 0.70, 0.64, 0.65, None],
    [0.39, 0.62, 0.75, 0.66, 0.56, 0.65, 0.47, None],
    [0.81, 0.48, 0.75, 0.50, 0.49, 0.49, 0.45, None],
    [0.63, 0.79, 0.67, 0.76, 0.74, 0.74, 0.71, None],
    [0.24, 0.76, 0.24, 0.76, 0.75, 0.76, 0.86, None],
    [0.25, 0.69, 0.24, 0.64, 0.86, 0.63, 0.70, None],
    [0.26, 0.74, 0.31, 0.70, 0.58, 0.65, 0.83, None],
    [0.33, 0.78, 0.33, 0.84, 0.50, 0.83, 0.65, 0.75],
    [0.28, 0.73, 0.33, 0.68, 0.84, 0.69, 0.80, 0.84],
    [0.84, 0.66, 0.31, 0.72, 0.85, 0.70, 0.79, 0.83],
    [0.76, 0.83, 0.35, 0.81, 0.70, 0.73, 0.77, 0.75],
    [0.64, 0.61, 0.58, 0.60, 0.62, 0.62, 0.89, 0.61],
    [0.60, 0.81, 0.20, 0.63, 0.62, 0.77, 0.61, 0.59],
    [0.63, 0.67, 0.27, 0.67, 0.61, 0.80, 0.45, 0.84],
    [None, None, None, None, None, None, None, None]
]

# Convert data to numpy array
data_array = np.array(data, dtype=float)
ytick_labels = [1, 2, 3, 4, 5, 6, 7, 1, 3, 5, 4, 2, 7, 6, 3, 1, 2, 7, 5, 8, 4, 5, 1, 4, 7, 2, 8, 3]
# Plot heatmap
plt.figure(figsize=(10, 6))
plt.imshow(data_array, cmap='viridis', interpolation='nearest')
plt.colorbar(label='Accuracy')
plt.xticks(np.arange(8), np.arange(1, 9))
plt.yticks(np.arange(len(ytick_labels)),ytick_labels)
plt.xlabel('Tasks')
plt.ylabel('Task order')
plt.title('Heatmap of Forgetting Metric')
plt.show()

