# -*- coding: utf-8 -*-
"""
This script is to convert features to descriptions for audio and facial modalities
"""

import os
import pickle
import numpy as np

data_root = '../Data/Hazumi'

save_root = '../Data/Hazumi'

feature2des = {}
with open('../Data_ori/Hazumi/feature_des.txt', 'r', encoding='utf-8') as f_in:
    lines = f_in.readlines()
for line in lines:
    feature2des[line.split('-')[0]] = line.strip('\n').split('-')[2]

with open(os.path.join(data_root, 'Pre_HazumiNeedContents_with_acoustic.pkl'), 'rb') as pkl_i:
    pre_dict = pickle.load(pkl_i)
with open(os.path.join(data_root, 'Pre_InfoHazumiNeedContents.pkl'), 'rb') as pkl_i:
    info_dict = pickle.load(pkl_i)

info_dict['feature2des'] = feature2des #info dict will contain feature2des, all variance-related values of modalities, 
                                       #label2id and id2label, and other statistics

feature_key_to_des_key = {'pitch': 'pitch_des', 
                          'energy': 'energy_des'}

#Hazumi is special on having data of each modality
#for example, some audio data was missing due to the missing of video; some annotations were missing due to the missing of video
#some AU was missing due to the time annotation or kinect missing
#not counted precisely but no more than 1% of original data was missing
#so in this experiment, these missing data of modality/annotation will be discarded

#find all index that no missing annotation/audio/AU
use_index_list = {}
for group in pre_dict:
    use_index_list[group] = []
    for index in pre_dict[group]['text']:
        if(pre_dict[group]['has_label'][index] == 0 or
           pre_dict[group]['has_AU'][index] == 0 or 
           pre_dict[group]['has_acoustic'][index] == 0):
            continue
        else:
            use_index_list[group].append(index)

#compute acoustic pitch and energy to determine four categories
#categories are determined based on the change of mean in three evenly divided periods in one utterance
#all wav files should be existed, so no _NONE_ case
for group in pre_dict:
    for key in ['AU_des_list', 'pitch_des', 'energy_des', 'AU_mean']:
        pre_dict[group][key] = {}
    #pitch and energy
    for feature_key in ['pitch', 'energy']:
        for index in use_index_list[group]:
            feature = pre_dict[group][feature_key][index]
            divide_interval = int(feature.shape[0] / 3)
            period_means = [np.mean(feature[:divide_interval]), 
                            np.mean(feature[divide_interval: 2 * divide_interval]), 
                            np.mean(feature[2 * divide_interval:])]
            
            if(feature_key == 'pitch'):
                des_head = 'Pitch '
            elif(feature_key == 'energy'):
                des_head = 'RMSE '
            
            #categories
            if(period_means[0] >= period_means[1] and period_means[1] > period_means[2]): #decrease from high to low
                pre_dict[group][feature_key_to_des_key[feature_key]][index] = des_head + '1'
            elif(period_means[0] > period_means[1] and period_means[1] >= period_means[2]): #not decrease then decrease is treated as decrease
                pre_dict[group][feature_key_to_des_key[feature_key]][index] = des_head + '1'
            elif(period_means[0] <= period_means[1] and period_means[1] < period_means[2]): #increase from low to high
                pre_dict[group][feature_key_to_des_key[feature_key]][index] = des_head + '2'
            elif(period_means[0] < period_means[1] and period_means[1] <= period_means[2]): #not increase then increase is treated as increase
                pre_dict[group][feature_key_to_des_key[feature_key]][index] = des_head + '2'
            elif(period_means[0] < period_means[1] and period_means[1] > period_means[2]): #rise then fall
                pre_dict[group][feature_key_to_des_key[feature_key]][index] = des_head + '3'
            elif(period_means[0] > period_means[1] and period_means[1] < period_means[2]): #fall then rise
                pre_dict[group][feature_key_to_des_key[feature_key]][index] = des_head + '4'
    
    #AU
    for index in use_index_list[group]:
        AU_mean = np.mean(pre_dict[group]['AU'][index], axis=0)
        pre_dict[group]['AU_mean'][index] = AU_mean
        pre_dict[group]['AU_des_list'][index] = []
        for AU_i, AU in enumerate(info_dict['AU_list']):
            if(AU_mean[AU_i] >= 0.5):
                pre_dict[group]['AU_des_list'][index].append(AU)

info_dict['des_sta'] = {}
for f_key in ['pitch_des', 'energy_des']:
    info_dict['des_sta'][f_key] = {}
    for group in pre_dict:
        for index in pre_dict[group][f_key]:
            if(pre_dict[group][f_key][index] not in info_dict['des_sta'][f_key].keys()):
                info_dict['des_sta'][f_key][pre_dict[group][f_key][index]] = 1
            else:
                info_dict['des_sta'][f_key][pre_dict[group][f_key][index]] += 1

info_dict['des_sta']['no_facial_sta'] = {}
info_dict['des_sta']['no_facial_sta']['NoAU'] = 0
info_dict['des_sta']['no_facial_sta']['ExpressAU'] = 0
for group in pre_dict:
    for index in pre_dict[group]['AU_des_list']:
        if(len(pre_dict[group]['AU_des_list'][index]) == 0):
            info_dict['des_sta']['no_facial_sta']['NoAU'] += 1
        else:
            info_dict['des_sta']['no_facial_sta']['ExpressAU'] += 1

#label2id and id2label
info_dict['label2id'] = {}
info_dict['id2label'] = {}
index = 0
for l in ['low', 'high']: #binary classification, <=4 is low, >4 is high
    if(l not in info_dict['label2id'].keys()):
        info_dict['label2id'][l] = index
        info_dict['id2label'][index] = l
        index += 1

#sort index and save to a new dict to output
#the reason not sort group here is that in future it may be necessary to use context (previous and later turns) for modeling
#if sort by index here, then that time the program must from this script
#if sort the index in the next script (s3) then the process will be from the Pre_data but not the pre-pre-data, seems to be more clear
#anyway, the group will be flatten in s3 to become a data dict contains [index][KEY] rather than [group][KEY][index] here
data_dict = {}
for group in pre_dict:
    data_dict[group] = {}
    d_index = 0
    for f_key in ['text', 'label_selfsentiment', 'label_thirdpartysentiment', 'pitch_des', 'energy_des', 'AU_des_list', 'pitch_des_text', 'energy_des_text', 'AU_des_list_text', 
                  'AU_mean', 'IS09', 'eGeMAPS', 
                  'original_index']: #original index can be used to select previous/later turns of a given utterance
        data_dict[group][f_key] = {}
    for index in use_index_list[group]:
        for f_key in ['pitch_des', 'energy_des', 'AU_des_list']:
            data_dict[group][f_key][d_index] = pre_dict[group][f_key][index]
            if(f_key in ['pitch_des', 'energy_des']):
                data_dict[group][f_key + '_text'][d_index] = info_dict['feature2des'][pre_dict[group][f_key][index]]
            if(f_key in ['AU_des_list']):
                data_dict[group][f_key + '_text'][d_index] = []
                for au in pre_dict[group][f_key][index]:
                    data_dict[group][f_key + '_text'][d_index].append(info_dict['feature2des'][au.split('_')[0]])
        for f_key in ['AU_mean', 'IS09', 'eGeMAPS']: #features for baseline model
            data_dict[group][f_key][d_index] = pre_dict[group][f_key][index]
        data_dict[group]['text'][d_index] = pre_dict[group]['text'][index]
        data_dict[group]['label_selfsentiment'][d_index] = pre_dict[group]['label_selfsentiment'][index]
        data_dict[group]['label_thirdpartysentiment'][d_index] = pre_dict[group]['label_thirdpartysentiment'][index]
        data_dict[group]['original_index'][d_index] = index
        d_index += 1

with open(os.path.join(save_root, 'Pre_DataDict.pkl'), 'wb') as pkl_o:
    pickle.dump(data_dict, pkl_o)
with open(os.path.join(save_root, 'Pre_InfoDict.pkl'), 'wb') as pkl_o:
    pickle.dump(info_dict, pkl_o)