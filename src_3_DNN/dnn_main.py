# -*- coding: utf-8 -*-
"""
This script is to run DNN-base and Early Fusion experiments
"""

import os
import sys
sys.path.append('../src_utils')
sys.path.append('../src_model')
import time
import pickle
import random
import numpy as np

import dnn

import torch
import torch.utils.data

import ex_config
import no_valid_ex_trainer
import has_valid_ex_trainer
import ex_process_dnn
from ex_config import config_vars

if(len(sys.argv) == 1):
    EX_GROUP = 'DNNEARLY'
    EX_DATASET = 'HazumiThird'
    HAS_VALID = True
    MODALITY = 'L+A+F'
    FOLD_TRAIN = True
    FOLDS = 5
    EPOCH = 200
    BATCH_SIZE = 16
    HIDDEN_SIZE = [256, 256]
    LR = 0.001
    TRAIN_PERCENT = 0.8
    
    TEST_FLAG = 1
    
    config_vars['ex_group'] = EX_GROUP
    config_vars['ex_dataset'] = EX_DATASET
    config_vars['lr'] = LR
    config_vars['batch_size'] = BATCH_SIZE
    config_vars['epoch'] = EPOCH
    config_vars['test_flag'] = TEST_FLAG
    config_vars['hidden_size'] = HIDDEN_SIZE
    config_vars['train_percent'] = TRAIN_PERCENT
    config_vars['fold_train'] = FOLD_TRAIN
    config_vars['folds'] = FOLDS
    config_vars['has_valid'] = HAS_VALID
    config_vars['modality'] = MODALITY

print(config_vars)

GPU_USE = torch.cuda.is_available()

SEED = random.randint(0, 1000)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

data_path = '../Data/' + config_vars['ex_dataset']

with open(os.path.join(data_path, 'Data_FreezeModelDataDict.pkl'), 'rb') as pkl_i:
    data_dict = pickle.load(pkl_i)
with open(os.path.join(data_path, 'Pre_InfoDict.pkl'), 'rb') as pkl_i:
    info_dict = pickle.load(pkl_i)

ex_info_dict = info_dict

save_root = '../results'
result_dir_head = 'HasValid_' if config_vars['has_valid'] == True else 'NoValid_'
save_path = os.path.join(save_root, result_dir_head + config_vars['ex_group'] + '_' + config_vars['ex_dataset'] + '_LR_' + str(config_vars['lr']) + \
                         '_Layer' + str(len(config_vars['hidden_size'])) + '_' + '-'.join([str(hs) for hs in config_vars['hidden_size']]))
time_stamp_str = ''
for t in time.localtime()[0: 6]:
    if(len(str(t)) <= 2):
        t = '%.2d' %t
    time_stamp_str += str(t)
if(config_vars['test_flag'] == 0):
    save_path = save_path + '_t' + time_stamp_str
if(os.path.exists(save_path) == False):
    os.mkdir(save_path)
#%%
model_key = 'DNN'

modality_all = list(data_dict['modality_text']['Bert'].keys())

data_dict['BERT'] = data_dict['freeze_model_repr']['Bert']['MEAN']['L']
modality2data_key = {'L': 'BERT', #Bert-CLS
                     'A': 'IS09', 
                     'F': 'AU_mean'}

#make fold group lists
fold_train_group = {}
if(config_vars['ex_dataset'][:6] == 'Hazumi'):
    group_list = []
    for index in data_dict['text']:
        if(data_dict['group'][index] not in group_list):
            group_list.append(data_dict['group'][index])
    fold_interval = int(len(group_list) / config_vars['folds'])
    for fold in range(config_vars['folds']):
        fold_train_group[fold + 1] = group_list[0: fold * fold_interval] + group_list[(fold + 1) * fold_interval:]

#%%
if(config_vars['modality'] == 'ALL'):
    modality_group = modality_all
else:
    if(config_vars['modality'] not in modality_all):
        raise(NameError('The set modality is not in experiment modalities'))
    else:
        modality_group = [config_vars['modality']]

for modality_key in modality_group:
    print(modality_key)
    input_size = 0
    for m in modality_key.split('+'):
        input_size += data_dict[modality2data_key[m]][0].shape[-1]
    
    result_save_path = os.path.join(save_path, modality_key)
    if(os.path.exists(result_save_path) == False):
        os.mkdir(result_save_path)
    all_fold_average_result_dict = {}
    if(config_vars['fold_train'] == True):
        fold_group = list(range(config_vars['folds']))
    elif(config_vars['fold_train'] == False):
        fold_group = [4] #the default setting is to use the last 20% data as test set, the first 80% as training set
    for fold in fold_group:
        #split fold data
        if(config_vars['has_valid'] == False):
            train_dataloader, test_dataloader, label_weight_tensor = ex_process_dnn.split_fold_data_no_valid(data_dict, 
                                                                                                         ex_info_dict, 
                                                                                                         config_vars,
                                                                                                         model_key, 
                                                                                                         modality_key, 
                                                                                                         modality2data_key, 
                                                                                                         fold, 
                                                                                                         fold_train_group=fold_train_group)
        elif(config_vars['has_valid'] == True):
            train_dataloader, valid_dataloader, test_dataloader, label_weight_tensor = ex_process_dnn.split_fold_data_has_valid(data_dict, 
                                                                                                                            ex_info_dict, 
                                                                                                                            config_vars,
                                                                                                                            model_key, 
                                                                                                                            modality_key, 
                                                                                                                            modality2data_key, 
                                                                                                                            fold, 
                                                                                                                            fold_train_group=fold_train_group)
                            
        configs = ex_config.configuration(input_size=input_size, #equal to Bert or RoBerta's output size, but not adapatble for others 
                                          linear_n_layers=len(config_vars['hidden_size']), 
                                          linear_middle_size=config_vars['hidden_size'], 
                                          batch_size=config_vars['batch_size'], 
                                          epoch=config_vars['epoch'], 
                                          ep_before_early_stop=config_vars['ep_before_early_stop'], 
                                          early_stop=config_vars['early_stop'], 
                                          ex_group=config_vars['ex_group'], 
                                          ex_dataset=config_vars['ex_dataset'], 
                                          pretrained_pooling=config_vars['pooling'], 
                                          lr=config_vars['lr'], 
                                          info_dict=ex_info_dict, 
                                          GPU_USE=GPU_USE, 
                                          modalities=modality_key, 
                                          modalities_save_path=result_save_path, 
                                          label_size=len(ex_info_dict['id2label']), 
                                          test_flag=config_vars['test_flag'], 
                                          fold_num=fold + 1, 
                                          random_seed=SEED)
        
        if(config_vars['ex_group'] == 'DNN'):
            model = dnn.DNN(configs)
        elif(config_vars['ex_group'] == 'DNNEARLY'):
            model = dnn.DNN_early_fusion_1(configs)
            
        if(configs.GPU_USE):
            model = model.cuda()
        if(config_vars['has_valid'] == False):
            result_dict = no_valid_ex_trainer.train_process(train_dataloader, test_dataloader, label_weight_tensor, model, ex_info_dict, 
                                              configs, 0, GPU_USE)
        elif(config_vars['has_valid'] == True):
            result_dict = has_valid_ex_trainer.train_process(train_dataloader, valid_dataloader, test_dataloader, label_weight_tensor, model, ex_info_dict, 
                                              configs, 0, GPU_USE)
            
        all_fold_average_result_dict[fold + 1] = result_dict
    aver_result_dict = {}
    for r_key in ['test_acc', 'test_weighted_f1', 'test_macro_f1']:
        metric_list = []
        for fold in fold_group:
            metric_list.append(all_fold_average_result_dict[fold + 1][r_key])
        aver_result_dict[r_key] = sum(metric_list) / len(metric_list)
    all_fold_average_result_dict['Aver'] = aver_result_dict
    with open(configs.all_fold_result_save_path, 'wb') as pkl_o:
        pickle.dump(all_fold_average_result_dict, pkl_o)
    with open(configs.aver_metric_result_save_path, 'w') as f_o:
        f_o.write('Average results: Acc : ' + str(round(aver_result_dict['test_acc'], 5)) + \
                  ' | Weighted F1 : ' + str(round(aver_result_dict['test_weighted_f1'], 5)) + \
                  ' | Macro F1 : ' + str(round(aver_result_dict['test_macro_f1'], 5)))
    if(config_vars['test_flag'] == 1):
        break