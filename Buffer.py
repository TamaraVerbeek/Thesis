# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:06:44 2023

@author: tamar
"""
import torch.nn as nn
import torch 
import numpy as np
import pdb


class Buffer(nn.Module):
    def __init__(self, args, input_size):
        super().__init__()
        self.args = args
        
        buffer_size = self.args.mem_size * 1
        bx = torch.FloatTensor(buffer_size, input_size).fill_(0)
        by = torch.LongTensor(buffer_size).fill_(0)
        
        logits = torch.FloatTensor(buffer_size, args.n_classes).fill_(0)
    
        self.current_index = 0
        self.n_seen_so_far = 0
        self.is_full       = 0
    
        self.register_buffer('bx', bx)
        self.register_buffer('by', by)
        self.register_buffer('logits', logits)
        
    def sample(self, amt):
        bx, by = self.bx[:self.current_index], self.by[:self.current_index]
        indices = torch.from_numpy(np.random.choice(bx.size(0), amt, replace=False))

        return bx[indices], by[indices]
    

    def add_reservoir(self, x, y, logits):
        n_elem = x.size(0)
        save_logits = logits is not None

        # add whatever still fits in the buffer
        place_left = max(0, self.bx.size(0) - self.current_index)
        if place_left:
            offset = min(place_left, n_elem)
            self.bx[self.current_index: self.current_index + offset].copy_(x[:offset].detach())
            self.by[self.current_index: self.current_index + offset].copy_(y[:offset].detach().view(-1))
            if save_logits:
                self.logits[self.current_index: self.current_index + offset].data.copy_(logits[:offset])

            self.current_index += offset
            self.n_seen_so_far += offset

            # everything was added
            if offset == x.size(0):
                return

        self.place_left = False

        # remove what is already in the buffer
        x, y = x[place_left:], y[place_left:]

        indices = torch.FloatTensor(x.size(0)).to(x.device).uniform_(0, self.n_seen_so_far).long()
        valid_indices = (indices < self.bx.size(0)).long()

        idx_new_data = valid_indices.nonzero().squeeze(-1)
        idx_buffer   = indices[idx_new_data]

        self.n_seen_so_far += x.size(0)

        if idx_buffer.numel() == 0:
            return

        assert idx_buffer.max() < self.bx.size(0), pdb.set_trace()
        assert idx_buffer.max() < self.by.size(0), pdb.set_trace()

        assert idx_new_data.max() < x.size(0), pdb.set_trace()
        assert idx_new_data.max() < y.size(0), pdb.set_trace()

        # perform overwrite op
        self.bx[idx_buffer] = x[idx_new_data]
        self.by[idx_buffer] = y[idx_new_data].view(-1).to(self.by.dtype)

        if save_logits:
            self.logits[idx_buffer] = logits[idx_new_data]