# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:19:43 2023

@author: 30816
"""

import os
import pickle

result_path = '../results_'

ex_dir = 'HasValid_Bert_Hazumi-Paragraph_LR_2e-05_PoolingMEAN_Layer1_768'

modality = 'L+A'

with open(os.path.join(result_path, ex_dir, modality, 'Results_f1.pkl'), 'rb') as pkl_i:
    result_dict = pickle.load(pkl_i)

with open(os.path.join('../Data', ex_dir.split('_')[2], 'Data_FreezeModelDataDict.pkl'), 'rb') as pkl_i:
    data_dict = pickle.load(pkl_i)