# -*- coding: utf-8 -*-
"""
This script is to run ChatGPT experiments
"""
import os
import sys
sys.path.append('../src_utils')
sys.path.append('../src_model')
import time
import tqdm
import pickle
import random

import openai

openai.api_key = '***'
gpt_model="gpt-3.5-turbo"

from ex_config import config_vars

if(len(sys.argv) == 1):
    EX_DATASET = 'Hazumi-ChatGPT'
    MODALITY = 'ALL'
    
    TEST_FLAG = 0
    
    config_vars['ex_dataset'] = EX_DATASET
    config_vars['modality'] = MODALITY
    config_vars['test_flag'] = TEST_FLAG

print(config_vars)

data_path = '../Data/' + config_vars['ex_dataset']
with open(os.path.join(data_path, 'Data_DataDictChatGPTText.pkl'), 'rb') as pkl_i:
    data_dict = pickle.load(pkl_i)

save_root = '../results'
result_dir_head = 'GiveExample' if config_vars['give_example'] == True else 'NoExample'
save_path = os.path.join(save_root, 'ChatGPT' + result_dir_head + '_' + config_vars['ex_dataset'])

time_stamp_str = ''
for t in time.localtime()[0: 6]:
    if(len(str(t)) <= 2):
        t = '%.2d' %t
    time_stamp_str += str(t)
if(config_vars['test_flag'] == 0):
    save_path = save_path + '_t' + time_stamp_str
if(os.path.exists(save_path) == False):
    os.mkdir(save_path)

if(config_vars['ex_group'] in ['Bert', 'FreezeBert']):
    model_key = 'Bert'
if(config_vars['ex_group'] in ['RoBerta', 'FreezeRoBerta']):
    model_key = 'RoBerta'

fold_train_group = {}
if(config_vars['ex_dataset'].split('-')[0] == 'Hazumi'):
    group_list = []
    for index in data_dict['text']:
        if(data_dict['group'][index] not in group_list):
            group_list.append(data_dict['group'][index])
    fold_interval = int(len(group_list) / config_vars['folds'])
    for fold in range(config_vars['folds']):
        fold_train_group[fold + 1] = group_list[0: fold * fold_interval] + group_list[(fold + 1) * fold_interval:]

modality_all = list(data_dict['modality_text']['Bert'].keys())

if(config_vars['modality'] == 'ALL'):
    modality_group = modality_all
else:
    if(config_vars['modality'] not in modality_all):
        raise(NameError('The set modality is not in experiment modalities'))
    else:
        modality_group = [config_vars['modality']]

for modality_key in modality_group:
    print('Start ', modality_key)
    response_dict = {}
    result_save_path = os.path.join(save_path, modality_key)
    if(os.path.exists(result_save_path) == False):
        os.mkdir(result_save_path)
    result_save_file = os.path.join(result_save_path, 'response.pkl')
    pbar = tqdm.tqdm(total=len(data_dict['chatgpt_des'][modality_key]), position=0)
    for index in data_dict['chatgpt_des'][modality_key]:
        text = data_dict['chatgpt_des'][modality_key][index]
        try: # sometimes there will be errors or timeout due to network or api process, so first try to request, if there is errors, then wait for some time then re-request
            response_json = openai.ChatCompletion.create(
                                          model=gpt_model,
                                          messages=[
                                                {"role": "user", "content": text},
                                                ]
                                        )
        except:
            print('Error occurred, wait 10 seconds.')
            time.sleep(10)
            response_json = openai.ChatCompletion.create(
                                          model=gpt_model,
                                          messages=[
                                                {"role": "user", "content": text},
                                                ]
                                        )
        response_dict[index] = {}
        response_dict[index]['content'] = response_json['choices'][0]['message']['content']
        response_dict[index]['role'] = response_json['choices'][0]['message']['role']
        
        pbar.update()
        time.sleep(random.randint(300, 400) / 100) #OpenAI limit rate is about 20/min, so make requests 10-15/min is safe, so here sleep for a while to make the request in the safe range
        
        if(config_vars['test_flag'] == 1):
            break
    if(config_vars['test_flag'] == 1):
        break
    with open(result_save_file, 'wb') as pkl_o:
        pickle.dump(response_dict, pkl_o)
    pbar.close()