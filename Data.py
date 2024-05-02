# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 13:56:26 2023

@author: Tamara Verbeek
"""

from Utils.LogFile import LogFile

class Data:
    def __init__(self, name, logfile):
        self.name = name
        self.logfile = logfile

        self.train = None
        self.test = None
        self.test_orig = None

        self.folds = None