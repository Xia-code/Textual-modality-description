# -*- coding: utf-8 -*-
"""
Created on Fri Mar  3 10:19:43 2023

@author: 30816
"""

import os
import pickle

data_path = '../Data/Hazumi-ChatGPT'
with open(os.path.join(data_path, 'Data_DataDictChatGPTText.pkl'), 'rb') as pkl_i:
    data_dict = pickle.load(pkl_i)
with open(os.path.join(data_path, 'Pre_InfoDict.pkl'), 'rb') as pkl_i:
    info_dict = pickle.load(pkl_i)