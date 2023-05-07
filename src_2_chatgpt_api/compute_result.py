# -*- coding: utf-8 -*-
"""
This script is to compute results from ChatGPT response
Orignally, this script was only run in IDE, you can add save outs to output results
the result is in the variable of 'result_dict'
"""

import os
import re
import random
import glob
import pickle
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

dataset = 'Hazumi-ChatGPT'
no_identification_choose_method = 'wrong'

result_path = '../results/ChatGPTNoExample_Hazumi-ChatGPT' # this is the result path
#result_path = glob.glob(result_find)[0]

with open(os.path.join('../Data/HazumiThird/Data_DataDictModalityText.pkl'), 'rb') as pkl_i:
    data_dict = pickle.load(pkl_i)
    
dataset_label_set = {'Hazumi-ChatGPT': ['高い', '低い']}
    
result_dict = {}
for key in ['ua', 'wa', 'macro_f1', 'weighted_f1', 'cm']:
    result_dict[key] = {}

no_identification_dict = {}
no_identification_dict_others = {}
japanese_comp_list = ['属していません', '属していない', '関連していません', '回答できません', 
                      '属さない', '判断できません', '把握できません']

modality_paths = glob.glob(os.path.join(result_path, '*'))
for modality_path in modality_paths:
    modality = modality_path.split('\\')[-1]
    no_identification_dict[modality] = {}
    no_identification_dict_others[modality] = {}
    identification_dict_temp = {}
    if(os.path.exists(os.path.join(modality_path, 'response.pkl')) == False):
        continue
    with open(os.path.join(modality_path, 'response.pkl'), 'rb') as pkl_i:
        response_dict = pickle.load(pkl_i)
    for index in response_dict:
        #for those indices that cannot identify the prediction by pre-defined patterns, exact the prediction mannually
        ###
        content = response_dict[index]['content'].lower()
        has_prediction_list = []
        for comp in dataset_label_set[dataset] + ['「低い」', '「高い」']:
            if(len(re.findall(comp, content)) > 0):
                has_prediction_list.append(comp)
        has_prediction_list = list(set(has_prediction_list))
        if(dataset.split('-')[0] == 'Hazumi'):
            if(len(has_prediction_list) > 1 or len(has_prediction_list) == 0):
                '''no_identification_flag = 0
                for comp in japanese_comp_list:
                    if(re.findall(comp, content) != []):
                        no_identification_flag = 1
                        break
                if(no_identification_flag == 1):
                    no_identification_dict[modality][index] = [index, response_dict[index]['content'].lower()]
                else:
                    no_identification_dict_others[modality][index] = [index, response_dict[index]['content'].lower()]'''
                no_identification_dict[modality][index] = [index, response_dict[index]['content'].lower()]
            elif(len(has_prediction_list) == 1):
                identification_dict_temp[index] = has_prediction_list[0]
    
    pred_list = []
    true_list = []
    for index in identification_dict_temp:
        if(dataset.split('-')[0] == 'Hazumi'):
            if(no_identification_choose_method == 'wrong'):
                if(index in no_identification_dict[modality]):
                    true_label = data_dict['label'][index]
                    for wrong_l in dataset_label_set[dataset]:
                        if(wrong_l != true_label):
                            sentiment = 'low' if wrong_l == '低い' else 'high'
                            pred_list.append(sentiment)
                            break
                else:
                    sentiment = 'low' if identification_dict_temp[index] == '低い' else 'high'
                    pred_list.append(sentiment)
            else:
                sentiment = 'low' if identification_dict_temp[index] == '低い' else 'high'
                pred_list.append(sentiment)
        true_emotion = data_dict['label'][index] if data_dict['label'][index] != 'exc' else 'hap'
        true_list.append(true_emotion)
    result_dict['ua'][modality] = accuracy_score(true_list, pred_list)
    result_dict['cm'][modality] = confusion_matrix(true_list, pred_list)
    emotion_acc_list = []
    for i, row in enumerate(result_dict['cm'][modality]):
        emotion_acc_list.append(result_dict['cm'][modality][i][i] / sum(result_dict['cm'][modality][i]))
    result_dict['wa'][modality] = sum(emotion_acc_list) / len(emotion_acc_list)
    result_dict['macro_f1'][modality] = f1_score(true_list, pred_list, average='macro')

print(result_dict) # added to print results