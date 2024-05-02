# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:55:50 2023

@author: Tamara Verbeek
"""
import torch
import torch.nn as nn
import copy

class E_Prompt(nn.Module):
    def __init__(self, num_heads, input_size, prompt_key_init, layers, prompt_init, length, prompt_prefix_size, prefix):
        super(E_Prompt, self).__init__()

        self.input_size = input_size
        self.num_heads = num_heads
        self.nr_of_eprompts = 1
        self.prompt_key_init = prompt_key_init
        self.key_storage = dict()
        self.layers = layers
        self.prompt_init = prompt_init
        self.task_storage = nn.ParameterDict()
        self.prefix_1_3 = nn.ParameterDict()
        self.prefix_4_6 = nn.ParameterDict()
        self.prefix_7_inf = nn.ParameterDict()
        self.prefix_0 = nn.ParameterDict()
        self.prompt = None
        self.size = length
        self.prompt_prefix_size = prompt_prefix_size 
        self.prefix_tune = prefix
        
    def init_e_prompt(self):
        print(f'Creating a new E-Prompt: {self.nr_of_eprompts}')
        if self.prefix_tune:
            dup = 2
        else:
            dup = 3
        E_task_shape = (dup, 1, self.num_heads, self.size, self.input_size // self.num_heads)
        prefix_length_shape = (dup, 1, self.num_heads, self.prompt_prefix_size, self.input_size // self.num_heads)
        if self.prompt_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(E_task_shape))
            self.prompt_1_5 = nn.Parameter(torch.zeros(prefix_length_shape))
            self.prompt_6_10 = nn.Parameter(torch.zeros(prefix_length_shape))
            self.prompt_11_15 = nn.Parameter(torch.zeros(prefix_length_shape))
            self.prompt_16_20 = nn.Parameter(torch.zeros(prefix_length_shape))
            self.prompt_21_inf = nn.Parameter(torch.randn(prefix_length_shape))
        elif self.prompt_init == 'uniform':
            self.prompt = nn.Parameter(torch.randn(E_task_shape)) 
            self.prompt_1_3 = nn.Parameter(torch.randn(prefix_length_shape))
            self.prompt_4_6 = nn.Parameter(torch.randn(prefix_length_shape))
            self.prompt_7_inf = nn.Parameter(torch.randn(prefix_length_shape))
            self.prompt_0 = nn.Parameter(torch.randn(prefix_length_shape))
            nn.init.uniform_(self.prompt, -1, 1)
            nn.init.uniform_(self.prompt_1_3, -1, 1)
            nn.init.uniform_(self.prompt_4_6, -1, 1)
            nn.init.uniform_(self.prompt_7_inf, -1, 1)
            nn.init.uniform_(self.prompt_0, -1, 1)
        self.prefix_1_3[str(self.nr_of_eprompts)] = self.prompt_1_3
        self.prefix_4_6[str(self.nr_of_eprompts)] = self.prompt_4_6
        self.prefix_7_inf[str(self.nr_of_eprompts)] = self.prompt_7_inf
        self.prefix_0[str(self.nr_of_eprompts)] = self.prompt_0
        self.task_storage[str(self.nr_of_eprompts)] = self.prompt
        return self.nr_of_eprompts

    def det_e_prompt_task(self, task):
        for key in self.task_storage.keys():
            if key == str(int(task)):
                return 
        self.nr_of_eprompts += 1
        key = self.init_e_prompt() 
    
    def get_e_prompt(self, i, bucket):
        if bucket == 0:
            bucket_prompt = self.prefix_0[i]
        elif bucket == None:
            return self.task_storage[i]
        elif bucket == 1:
            bucket_prompt = self.prefix_1_3[i]
        elif bucket == 2:
            bucket_prompt = self.prefix_4_6[i]
        else:
            bucket_prompt = self.prefix_7_inf[i]
        total_prompt = torch.cat([self.task_storage[i], bucket_prompt], dim = 3)
        return total_prompt

    def set_e_prompt(self, i, weights):
        self.task_storage[i] = weights
        