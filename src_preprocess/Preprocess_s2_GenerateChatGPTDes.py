# -*- coding: utf-8 -*-
"""
This script is for generating ChatGPT input based on paragraph construction descriptions
"""
import os
import pickle
import numpy as np

dataset = {1: 'Hazumi'}

dataset_choose = 1

data_root = '../Data/' + dataset[dataset_choose] + '-Paragraph'
save_root = '../Data/' + dataset[dataset_choose] + '-ChatGPT'

with open(os.path.join(data_root, 'Data_DataDictModalityText.pkl'), 'rb') as pkl_i:
    data_dict = pickle.load(pkl_i)
with open(os.path.join(data_root, 'Pre_InfoDict.pkl'), 'rb') as pkl_i:
    info_dict = pickle.load(pkl_i)

dataset_head_text = {1: 'という説明が与えられた。'}
dataset_label_text = {1: '[高い、低い]というセンチメントのカテゴリーが与えられた。'}
dataset_ask_text = {1: '先の説明はどのカテゴリーに属しているか？'}

chatgpt_des = {}
for modality in data_dict['modality_text']['Bert']:
    chatgpt_des[modality] = {}
    for index in data_dict['modality_text']['Bert'][modality]:
        if(dataset_choose == 1):
            chatgpt_des[modality][index] = '"' + data_dict['modality_text']['Bert'][modality][index] + '"' + dataset_head_text[dataset_choose]
            chatgpt_des[modality][index] += dataset_label_text[dataset_choose]
            chatgpt_des[modality][index] += dataset_ask_text[dataset_choose]

data_dict['chatgpt_des'] = chatgpt_des
with open(os.path.join(save_root, 'Data_DataDictChatGPTText.pkl'), 'wb') as pkl_o:
    pickle.dump(data_dict, pkl_o)