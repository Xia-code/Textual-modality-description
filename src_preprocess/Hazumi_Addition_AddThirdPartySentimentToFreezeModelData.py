# -*- coding: utf-8 -*-
"""
This script is to add third-party sentiment labels in data
"""
import os
import re
import pickle
import numpy as np

data_root = '../Data/Hazumi-Paragraph'

with open(os.path.join(data_root, 'Data_FreezeModelDataDict.pkl'), 'rb') as pkl_i:
    data_dict = pickle.load(pkl_i)

with open(os.path.join(data_root, 'Data_DataDictModalityText.pkl'), 'rb') as pkl_i:
    modality_dict = pickle.load(pkl_i)

data_dict['label_thirdpartysentiment'] = modality_dict['label_thirdpartysentiment']
with open(os.path.join(data_root, 'Data_FreezeModelDataDict.pkl'), 'wb') as pkl_o:
    pickle.dump(data_dict, pkl_o)