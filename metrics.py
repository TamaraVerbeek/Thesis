# -*- coding: utf-8 -*-
"""
Created on Wed Jan 17 15:54:47 2024

@author: tamar
"""
import pandas as pd

output = pd.read_csv('output/IOR parameters/window_size/output_window_350.csv')
concept_drift =  [5224, 10469, 15674, 21002, 26146, 31429, 36591, 41853, 47035] #IOR

valleys = [(5599, 0.685714286), (10804, 0.768571429), (1630, 0.705714286)]
    
concept_drift =  [0, 5224, 10469, 15674, 21002, 26146, 31429, 36591, 41853, 47035, 52324] #IOR

def calculate_average_between_indices(values, indices):
    averages = []

    for i in range(len(indices) - 1):
        start_index = indices[i]
        end_index = indices[i + 1]
        
        if start_index >= end_index:
            raise ValueError("Indices should be in increasing order.")

        subset = values[start_index + 1:end_index]
        avg = sum(subset) / len(subset) if len(subset) > 0 else 0
        averages.append(avg)

    return averages

calculate_average_between_indices(output['1.0'], concept_drift)