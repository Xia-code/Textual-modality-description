# -*- coding: utf-8 -*-
"""
This script is to build separator concatenation description inputs for experiments
"""

import os
import re
import pickle

data_root = '../Data/Hazumi'

with open(os.path.join(data_root, 'Pre_DataDict.pkl'), 'rb') as pkl_i:
    pre_dict = pickle.load(pkl_i)
with open(os.path.join(data_root, 'Pre_InfoDict.pkl'), 'rb') as pkl_i:
    info_dict = pickle.load(pkl_i)

modality2des = {'L': ['text_processed'], 
                'A': ['pitch_des_text', 'energy_des_text'], 
                'F': ['AU_des_list_text']}

modality_groups = ['L', 'L+A', 'L+F', 'L+A+F', 
                   'A', 'A+F', 'F']

#First flatten group into index
data_dict = {}
for f_key in list(pre_dict[list(pre_dict.keys())[0]].keys()) + ['group']:
    data_dict[f_key] = {}

d_index = 0
for group in pre_dict:   
    for index in pre_dict[group][list(pre_dict[group].keys())[0]]:
        for f_key in data_dict:
            if(f_key not in ['group']):
                data_dict[f_key][d_index] = pre_dict[group][f_key][index]
        data_dict['group'][d_index] = group
        d_index += 1

#modify text to remove (F xx) and replace | by comma
data_dict['text_processed'] = {}
for index in data_dict['text']:
    text = data_dict['text'][index].replace('|', '、')
    comp = r'(\(.*?\))'
    filers = re.findall(comp, text)
    for filer in filers:
        text = text.replace(filer, '')
    text = text.strip(', ')
    if(text == ''):
        text = ' '
    data_dict['text_processed'][index] = text

#concatenate order: basically A, F, L
modality_order = {0: 'A', 1: 'F', 2: 'L'}

separators = {'Bert': '[SEP]', 'RoBerta': '[SEP]'}

modality_text_dict = {}

for model_use in ['Bert', 'RoBerta']:
    modality_text_dict[model_use] = {}
    for modality_group in modality_groups:
        modality_text_dict[model_use][modality_group] = {}
        modalities = modality_group.split('+')
        for index in data_dict['text']:
            modality_text = ''
            for m_i in range(3):
                if modality_order[m_i] in modalities:
                    if(modality_order[m_i] in ['L', 'A']):
                        for des in modality2des[modality_order[m_i]]:
                            modality_text += data_dict[des][index]
                            modality_text += separators[model_use]
                    if(modality_order[m_i] in ['F']):
                        if(len(data_dict['AU_des_list_text'][index]) == 0):
                            modality_text += info_dict['feature2des']['NoObviousAU']
                        else:
                            for au in data_dict['AU_des_list_text'][index]:
                                modality_text += au
                                modality_text += separators[model_use]
                        
            modality_text_dict[model_use][modality_group][index] = modality_text.rstrip(separators[model_use])

data_dict['modality_text'] = modality_text_dict

#generate high/low label
data_dict['label'] = {}
data_dict['label_processed'] = {}
for index in data_dict['label_selfsentiment']:
    #some label is not well formatted, such as '6 delete', ' 4', so strip the space ' ' first, then choose the 0 index, this will choose the exact number
    data_dict['label'][index] = 'high' if int(data_dict['label_selfsentiment'][index].strip(' ')[0]) > 4 else 'low'
for index in data_dict['label']:
    data_dict['label_processed'][index] = info_dict['label2id'][data_dict['label'][index]]

with open(os.path.join(data_root, 'Data_DataDictModalityText.pkl'), 'wb') as pkl_o:
    pickle.dump(data_dict, pkl_o)