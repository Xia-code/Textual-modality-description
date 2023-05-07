# -*- coding: utf-8 -*-
"""
This script is to add audio features, and justify whether the feature of each audio segment was extracted correctly or not
if not, it means the audio has some problems, or the utterance were missing, or the video did not include such an utterance (usually happened on last few utterances in a dialogue, where the video was not recorded to the end)
we will discard the corresponding utterance in subsequential process
"""

import os
import csv
import pickle
import numpy as np

data_root = '../Data/Hazumi'
pitch_data_root = '../Data_ori/Hazumi/Matlab_Pitch'
energy_data_root = '../Data_ori/Hazumi/Matlab_RMS_Energy'
IS09_data_root = '../Data_ori/Hazumi/IS09'
eGeMAPS_data_root = '../Data_ori/Hazumi/eGeMAPS'

with open(os.path.join(data_root, 'Pre_HazumiNeedContents.pkl'), 'rb') as pkl_i:
    data_dict = pickle.load(pkl_i)

no_acoustic_count = 0

for speaker in data_dict:
    data_dict[speaker]['has_acoustic'] = {} #some wav/video missed contents, so no audio features were extracted. This kind of data will be discard in experiments.
    for key in ['pitch', 'energy', 'IS09', 'eGeMAPS']:
        data_dict[speaker][key] = {}
    #read in pitch and energy features
    group = speaker[:4]
    for index in data_dict[speaker]['text']:
        pitch_csv_file = os.path.join(pitch_data_root, group, speaker, speaker + '_' + str(index) + '.csv')
        energy_csv_file = os.path.join(energy_data_root, group, speaker, speaker + '_' + str(index) + '.csv')
        IS09_csv_file = os.path.join(IS09_data_root, group, speaker, speaker + '_' + str(index) + '.csv')
        eGeMAPS_csv_file = os.path.join(eGeMAPS_data_root, group, speaker, speaker + '_' + str(index) + '.csv')
        if(os.path.exists(pitch_csv_file) == False or
           os.path.exists(energy_csv_file) == False or 
           os.path.exists(IS09_csv_file) == False or 
           os.path.exists(eGeMAPS_csv_file) == False):
            data_dict[speaker]['has_acoustic'][index] = 0
            no_acoustic_count += 1
        else:
            with open(pitch_csv_file, 'r') as csv_i:
                csv_reader = csv.reader(csv_i)
                for row in csv_reader:
                    pitch_features = row
                    break #only one line, but sometimes the reader may read following blank lines, so break to avoid that situation
            
            with open(energy_csv_file, 'r') as csv_i:
                csv_reader = csv.reader(csv_i)
                for row in csv_reader:
                    energy_features = row
                    break
            
            with open(IS09_csv_file, 'r') as csv_i:
                csv_reader = csv.reader(csv_i)
                for row_i, row in enumerate(csv_reader):
                    if(row_i == 391):
                        IS09_feature = row[1: -1]
            
            with open(eGeMAPS_csv_file, 'r') as csv_i:
                csv_reader = csv.reader(csv_i)
                for row_i, row in enumerate(csv_reader):
                    if(row_i == 95):
                        eGeMAPS_feature = row[1: -1]
            
            add_flag = 1
            if(len(pitch_features) == 1 or len(energy_features) == 1):
                data_dict[speaker]['has_acoustic'][index] = 0
                no_acoustic_count += 1
                add_flag = 0
            elif(len(np.unique(np.array(pitch_features, dtype='float'))) == 1):
                data_dict[speaker]['has_acoustic'][index] = 0
                no_acoustic_count += 1
                add_flag = 0
            
            if(len(IS09_feature) < 384 or len(eGeMAPS_feature) < 88):
                data_dict[speaker]['has_acoustic'][index] = 0
                no_acoustic_count += 1
                add_flag = 0
            
            if(add_flag == 1):
                data_dict[speaker]['pitch'][index] = np.array(pitch_features, dtype='float')
                data_dict[speaker]['energy'][index] = np.array(energy_features, dtype='float')
                data_dict[speaker]['IS09'][index] = np.array(IS09_feature, dtype='float')
                data_dict[speaker]['eGeMAPS'][index] = np.array(eGeMAPS_feature, dtype='float')
                data_dict[speaker]['has_acoustic'][index] = 1
            
    print(speaker + ' Over')

with open(os.path.join(data_root, 'Pre_HazumiNeedContents_with_acoustic.pkl'), 'wb') as pkl_o:
    pickle.dump(data_dict, pkl_o)