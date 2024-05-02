# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:53:07 2023

@author: Taqmara Verbeek
"""
import pandas as pd
import numpy as np
import os
from Data import Data
from Utils.LogFile import LogFile
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

class Preprocess:
    def __init__(self, args):
        self.data = None
        self.args = args
        self.out_encoder = OneHotEncoder(sparse=False)
        self.one_encoder_event = OneHotEncoder(sparse=False)
        self.one_encoder_resource = OneHotEncoder(sparse=False)
        self.categories_event = None
        self.categories_resource = None
        self.k = args.prefix_length
        self.trace = 'case'
        self.contextdata = None
        
    def get_data(self):
        path = os.path.join(self.args.data_path, self.args.dataset)
        print(path)
        d = Data(self.args.dataset,
                 LogFile(filename=path, delim=",", header=0, rows=10000, time_attr="completeTime", trace_attr="case",
                         activity_attr='event', convert=False))
        d.logfile.keep_attributes(['event', 'completeTime'])
        listOfevents = [[value] for value in list(d.logfile.data['event'])]
        listOfevents.append(['0'])
        print('listOfevents')
        self.out_encoder.fit(listOfevents)
        self.one_encoder_event.fit(listOfevents)
        self.categories_event =  [1] * len(set([value for value in list(d.logfile.data['event'])])) #self.args.nrOfEvents
        print(len(self.categories_event))
        return d, len(self.categories_event)
    
    def sublists(self, my_dict):
        return list(map(list, zip(*my_dict.values())))

    def get_all_previous(self, contextdata):
        previous = {}
        for i in range(0, self.k):    
            print(np.array(self.contextdata['event_Prev'+str(i)]).reshape(-1,1))
            previous['prev'+str(i)] = list(self.one_encoder_event.transform(np.array(self.contextdata['event_Prev'+str(i)]).reshape(-1,1)))
        lists = self.sublists(previous)
        return lists

    def create_k_context(self):
        """
        Create the k-context from the current LogFile

        :return: None
        """
        print("Create k-context:", self.k)

        if self.k == 0:
            self.contextdata = self.get_data()

        if self.contextdata is None:
            result = map(self.create_k_context_trace, self.get_data()[0].logfile.data.groupby([self.trace], sort= False))
            self.contextdata = pd.concat(result, ignore_index=True)
        lists = self.get_all_previous(self.contextdata)
        
        targets = list(self.one_encoder_event.transform(np.array(self.contextdata['event']).reshape(-1,1)))
        return lists, self.get_data()[0].logfile.data, targets, len(self.categories_event)

    def create_k_context_trace(self, trace):
        contextdata = pd.DataFrame()

        trace_data = trace[1]
        shift_data = trace_data.shift().fillna('0')
        shift_data.at[shift_data.first_valid_index(), self.trace] = trace[0]
        joined_trace = shift_data.join(trace_data, lsuffix="_Prev0")
        for i in range(1, self.k):
            shift_data = shift_data.shift().fillna('0')
            shift_data.at[shift_data.first_valid_index(), self.trace] = trace[0]
            joined_trace = shift_data.join(joined_trace, lsuffix="_Prev%i" % i)
        contextdata = pd.concat([joined_trace, contextdata], ignore_index = True)
        contextdata = contextdata.astype("str", errors="ignore")
        return contextdata
    
    """
    def add_data(self, window_case, length, data):
        df = pd.DataFrame(columns = list(data.index))
        for l in range(length):
            index = len(df)
            df.loc[index, 'event'] = 'none'
#            df.loc[index, 'role'] = 0
            df.loc[index, 'task'] = data['task']
            df.loc[index, 'case'] = data['case']
        #if we did have samples in our window of this case we add them
        if len(window_case) != 0:
            if len(window_case) == 10:
                #df = window_case[['event', 'role', 'completeTime', 'task', 'case']].reset_index()
                df = window_case[['event','completeTime', 'task', 'case']].reset_index()
            else:
                for ind, w in window_case.iterrows():
                    df.loc[length+ind] = w
        df.loc[len(df)] = data
        return df
    
    def get_prefix(self, data):
        inp = []
        for _,d in data.iloc[:-1].iterrows():
            inp.extend(self.one_encoder_event.transform(np.array(d['event']).reshape(1,-1)))
#            inp.extend(self.one_encoder_resource.transform(np.array(d['role']).reshape(1,-1)))
        return inp
    
    
    def encode(self, data):
        #data = data[['event', 'role',  'task']]
        data = data[['event',  'task']]
        prefix = self.get_prefix(data)
        #prefix = self.concatenate_adjacent_lists(prefix)
        target = data.tail(1)['event']
        target = self.out_encoder.transform(np.array(target).reshape(1,-1))[0]
        task = int(list(set(data['task']))[0])
        return prefix, target, task
    
    def get_encoding(self, args, window, d):
        #if we haven't seen this case id yet
        if len(window) == 0:
            d = self.add_data(pd.DataFrame(window), args.prefix_length, d)
            prefix, target, task = self.encode(d)
            return prefix, target, task
        
        #get all samples from the window that belong to the case id
        window_case = pd.DataFrame(window)[pd.DataFrame(window)['case'] == d['case']]
        if not window_case.empty:
            window_case = window_case[window_case.index < d.name].reset_index()
        #if we have seen this case id but don't have enough samples for an entire prefix
        if len(window_case) < args.prefix_length:
            length = args.prefix_length - len(window_case)
            d = self.add_data(window_case, length, d)
            prefix, target, task = self.encode(d)
        #if we have seen at least 10 samples of this case id
        else:
            last_samples_window_case = window_case.tail(args.prefix_length)
            d = self.add_data(last_samples_window_case, 0, d)
            prefix, target, task = self.encode(d)
        return prefix, target, task
        
    def get_encoding_new(self, args, d, ind, data):
        window = args.window_size
        #if we haven't seen this case id yet
        if ind == 0:
            d = self.add_data(pd.DataFrame(data.loc[0:ind-1]), args.prefix_length, d)
            prefix, target, task = self.encode(d)
            return prefix, target, task
        elif ind < window:
            #get all samples from the seen data that belong to the case id
            window_case = pd.DataFrame(data.loc[0:ind])[pd.DataFrame(data.loc[0:ind])['case'] == d['case']]
        else:
            #get all samples from the window that belong to the case id
            window_case = pd.DataFrame(data.loc[(ind-window):ind])[pd.DataFrame(data.loc[(ind-window):ind])['case'] == d['case']]
        
        if not window_case.empty:
            window_case = window_case[window_case.index < d.name].reset_index()
        #if we have seen this case id but don't have enough samples for an entire prefix
        if len(window_case) < args.prefix_length:
            length = args.prefix_length - len(window_case)
            d = self.add_data(window_case, length, d)
            prefix, target, task = self.encode(d)
        #if we have seen at least 10 samples of this case id
        else:
            last_samples_window_case = window_case.tail(args.prefix_length)
            d = self.add_data(last_samples_window_case, 0, d)
            prefix, target, task = self.encode(d)
        return prefix, target, task
    """