# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:12:37 2023

@author: tamar
"""
import pandas as pd
from pm4py.objects.log.importer.xes import importer as xes_importer
from pm4py.streaming.importer.xes import importer as stream_importer
from pm4py.objects.conversion.log import converter
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import DataLoader
from SDL import CustomDataset
import numpy as np 
import torch
from itertools import chain

class transform():
    def __init__(self, data):
        self.data, self.classes, self.possible_activities = self.import_xes_next_event(data)
        self.matrix = pd.DataFrame(columns = self.possible_activities)
        self.encoder = OneHotEncoder(categories = self.possible_activities)

    def import_xes_next_event(self, data):
        classes = data.drop_duplicates(subset='next_activity')
        classes = classes['next_activity']
        
        pos_activities = data['concept:name'].unique()
        
        return data, classes, pos_activities 

    def one_hot_encoding_sdl(self, df, size):
        y = []
        j = 0
        #df = df.data
        for i in range(size-1, len(df)):
            if len(set(df.iloc[i-9:i]['case:concept:name'])) == 1:
                new = [0] * len(self.matrix.columns)
                self.matrix.loc[j] = new
                activities = list(set(df.iloc[i-9:i]['concept:name']))
                self.matrix.loc[j, activities] = 1
                j += 1
                y.append(df.iloc[i]['next_activity'])
        return self.matrix, y
    
    def freq_encoding_sdl(self, df, size):
        df = df.data
        y = []
        j = 0
        for i in range(size-1, len(df)):
            if len(set(df.iloc[i-9:i]['case:concept:name'])) == 1:
                new = [0] * len(self.matrix.columns)
                self.matrix.loc[j] = new
                activities = list(df.loc[i-9:i,'concept:name'])
                for act in activities:
                    self.matrix.loc[j, act] += 1
                j += 1
                y.append(df.loc[i, 'next_activity'])
        return self.matrix, y
    
    def create_train_set(self, inp, out, batch_size, shuffle):
        encoding = []
        for column_name, column_data in inp.iterrows():
            all_encs = []
            for c in column_data:
                empty = np.zeros(len(self.possible_activities))
                empty[c-1] = 1
                all_encs.append(empty)
            encoding.append(list(chain(*all_encs)))
        one_hot_encoding = torch.tensor(encoding, dtype = torch.float32, requires_grad=True)

        #encoded = self.label_encoder.transform(out)
        #encoded_dict = {'encoded output': encoded}
        #encoded_df = pd.DataFrame(encoded_dict)
        encoded_y = torch.tensor(np.array(out))
        train_loader = DataLoader(CustomDataset(one_hot_encoding, encoded_y), batch_size, shuffle)
        return train_loader