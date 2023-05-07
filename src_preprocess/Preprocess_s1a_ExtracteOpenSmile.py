# -*- coding: utf-8 -*-
"""
This script is to extract opensmile feature sets for the Hazumi dataset
"""
import os
import glob

data_root = '../Data_ori/Hazumi/AudioSeg'

save_root_all = {1: '../Data_ori/Hazumi/IS09', 
                 2: '../Data_ori/Hazumi/eGeMAPS'}

feature_set_all = {1: 'IS09', 
                   2: 'eGeMAPS'}
open_smile_configs = {1: 'C:\\Users\\30816\\opensmile-3.0-win-x64\\config\\is09-13\\IS09_emotion.conf', 
                      2: 'C:\\Users\\30816\\opensmile-3.0-win-x64\\config\\egemaps\\v01a\\eGeMAPSv01a.conf'}

for feature_choose in open_smile_configs:
    feature_set = feature_set_all[feature_choose]
    open_smile_config = open_smile_configs[feature_choose]
    save_root = save_root_all[feature_choose]
    for group in ['1902', '1911']:
        if(os.path.exists(os.path.join(save_root, group)) == False):
            os.mkdir(os.path.join(save_root, group))
        speaker_dirs = glob.glob(os.path.join(data_root, group, '*'))
        for speaker_dir in speaker_dirs:
            speaker_name = speaker_dir.split('\\')[-1]
            if(os.path.exists(os.path.join(save_root, group, speaker_name)) == False):
                os.mkdir(os.path.join(save_root, group, speaker_name))
            wav_file_all = glob.glob(os.path.join(speaker_dir, '*.wav'))
            for wav_file in wav_file_all:
                wav_name = wav_file.split('\\')[-1].split('.')[0]
                csv_file = os.path.join(save_root, group, speaker_name, wav_name + '.csv')
                os.system('SMILExtract -C ' + open_smile_config + ' -I ' + wav_file + ' -O ' + csv_file)
            print(feature_set_all[feature_choose], group, speaker_name, 'Over')