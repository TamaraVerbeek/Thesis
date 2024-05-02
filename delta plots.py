# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 23:10:25 2024

@author: tamar
"""
import pandas as pd
import numpy as np

out1 = pd.read_csv('output/random_dataset/Dualprompt/output_dualprompt_10_5.csv')
out2 = pd.read_csv('output/random_dataset/Dualprompt/output_dualprompt_with_task_det.csv')

x1 = out1.iloc[:,0]
y1 = out1.iloc[:,1]
x2 = out2.iloc[:,0]
y2 = out2.iloc[:,1]

c = [2724, 5756, 9074, 12025, 15297, 18042, 20680, 23385, 26816, 30068, 33105, 36101, 38786, 41614, 44979, 47623, 50125, 52627, 55129, 57631, 60133, 63493, 66198, 69256, 72060, 75121, 77904]

print(np.mean(y1.iloc[c[18]:c[19]]))

print(np.mean(y2.iloc[c[18]:c[19]]))

out1 = pd.read_csv('output/BPIC15/output_dualprompt_10_5.csv')
out2 = pd.read_csv('output/random_dataset/Dualprompt/output_dualprompt_with_task_det.csv')

x1 = out1.iloc[:,0]
y1 = out1.iloc[:,1]
x2 = out2.iloc[:,0]
y2 = out2.iloc[:,1]

c = [2724, 5756, 9074, 12025, 15297, 18042, 20680, 23385, 26816, 30068, 33105, 36101, 38786, 41614, 44979, 47623, 50125, 52627, 55129, 57631, 60133, 63493, 66198, 69256, 72060, 75121, 77904]

print(np.mean(y1.iloc[c[18]:c[19]]))

print(np.mean(y2.iloc[c[18]:c[19]]))
