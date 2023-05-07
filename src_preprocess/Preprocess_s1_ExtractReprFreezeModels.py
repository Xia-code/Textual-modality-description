# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 17:43:56 2023

@author: 30816
"""

import os
import pickle
import torch
from transformers import RobertaTokenizer, RobertaModel, BertTokenizer, BertModel, T5Tokenizer, BertJapaneseTokenizer
from tqdm import tqdm
import numpy as np

dataset = {1: 'IEMOCAP', 
           2: 'Hazumi'}

dataset_models = {'Bert': {1: 'bert-base-uncased', 
                           2: 'cl-tohoku/bert-base-japanese'}, 
                  'RoBerta': {1: 'roberta-base', 
                              2: 'rinna/japanese-roberta-base'}}

data_choose = 2

data_variety = '-Paragraph' # '', '-LessSep', '-6-LessSep'

data_root = '../Data/' + dataset[data_choose] + data_variety

with open(os.path.join(data_root, 'Data_DataDictModalityText.pkl'), 'rb') as pkl_i:
    data_dict = pickle.load(pkl_i)

cuda_available = torch.cuda.is_available()

for key in dataset_models:
    print(key, ':', dataset_models[key][data_choose])

repr_dict = {}
for model_use in data_dict['modality_text']:
    repr_dict[model_use] = {}
    for repr_com in ['CLS', 'MEAN']:
        repr_dict[model_use][repr_com] = {}
    if(model_use == 'RoBerta'):
        if(dataset[data_choose] in ['IEMOCAP']):
            tokenizer = RobertaTokenizer.from_pretrained(dataset_models[model_use][data_choose])
            model = RobertaModel.from_pretrained(dataset_models[model_use][data_choose])
        elif(dataset[data_choose] in ['Hazumi']):
            tokenizer = T5Tokenizer.from_pretrained(dataset_models[model_use][data_choose])
            model = RobertaModel.from_pretrained(dataset_models[model_use][data_choose])
    elif(model_use == 'Bert'):
        if(dataset[data_choose] in ['IEMOCAP']):
            tokenizer = BertTokenizer.from_pretrained(dataset_models[model_use][data_choose])
            model = BertModel.from_pretrained(dataset_models[model_use][data_choose])
        elif(dataset[data_choose] in ['Hazumi']):
            tokenizer = BertJapaneseTokenizer.from_pretrained(dataset_models[model_use][data_choose], 
                                                              mecab_kwargs={'mecab_dic': 'unidic'})
            model = BertModel.from_pretrained(dataset_models[model_use][data_choose])
    if(cuda_available):
        model = model.cuda()
    for modality_group in data_dict['modality_text'][model_use]:
        for repr_com in ['CLS', 'MEAN']:
            repr_dict[model_use][repr_com][modality_group] = {}
        pbar = tqdm(total=len(data_dict['modality_text'][model_use][modality_group]), position=0)
        print(model_use, modality_group)
        for index in data_dict['modality_text'][model_use][modality_group]:
            text = data_dict['modality_text'][model_use][modality_group][index]
            encoded_input = tokenizer(text, return_tensors='pt')
            if(cuda_available):
                for key in encoded_input:
                    encoded_input[key] = encoded_input[key].cuda()
            model_output_all = model(**encoded_input, output_hidden_states=True)
            last_hidden_state = model_output_all['last_hidden_state'].detach().cpu().numpy() if cuda_available else model_output_all['last_hidden_state'].detach().numpy()
            repr_dict[model_use]['CLS'][modality_group][index] = last_hidden_state[:, 0, :]
            repr_dict[model_use]['MEAN'][modality_group][index] = np.mean(last_hidden_state, axis=1)
            pbar.update()
pbar.close()

data_dict['freeze_model_repr'] = repr_dict
with open(os.path.join(data_root, 'Data_FreezeModelDataDict.pkl'), 'wb') as pkl_o:
    pickle.dump(data_dict, pkl_o)