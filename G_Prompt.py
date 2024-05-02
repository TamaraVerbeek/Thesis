# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 10:55:50 2023

@author: Tamara Verbeek
"""
import torch
import torch.nn as nn
import copy

class G_Prompt(nn.Module):
    def __init__(self, num_heads, input_size, layers, prompt_init, length, prefix):
        super(G_Prompt, self).__init__()

        self.input_size = input_size
        self.num_heads = num_heads
        self.layers = layers
        self.prompt_init = prompt_init
        self.storage = nn.ParameterDict()
        self.prompt = None
        self.size = length
        self.prefix_tune = prefix
        
    def init_g_prompt(self):
        print('Initializing the G-Prompt')
        if self.prefix_tune:
            dup = 2
        else:
            dup = 3
        prompt_pool_shape = (dup, 1, self.num_heads, self.size, self.input_size // self.num_heads)
        if self.prompt_init == 'zero':
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif self.prompt_init == 'uniform':
            self.prompt = nn.Parameter(torch.randn(prompt_pool_shape)) # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
            nn.init.uniform_(self.prompt, -1, 1)
        self.storage['0'] = self.prompt
        return 
      
    
    def get_g_prompt(self):
        return self.storage['0']
    
   