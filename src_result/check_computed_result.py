# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:19:43 2023

@author: 30816
"""

import os
import pickle

result_path = '../result_sta'

result_file = 'results_'

with open(os.path.join(result_path, result_file + '.pkl'), 'rb') as pkl_i:
    result_dict = pickle.load(pkl_i)