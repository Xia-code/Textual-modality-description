# -*- coding: utf-8 -*-
"""
This script is to generate data for exepriments using third-party sentiment label
"""
import os
import pickle

data_root = '../Data'
change_data_names = {1: 'Hazumi', 
                     2: 'Hazumi-Paragraph'}
change_to_data_names = {1: 'HazumiThird', 
                        2: 'HazumiThird-Paragraph'}

data_choose = 1

with open(os.path.join(data_root, change_data_names[data_choose], 'Pre_InfoDict.pkl'), 'rb') as pkl_i:
    info_dict = pickle.load(pkl_i)

for data_file in ['Data_DataDictModalityText.pkl', 'Data_FreezeModelDataDict.pkl']:
    with open(os.path.join(data_root, change_data_names[data_choose], data_file), 'rb') as pkl_i:
        data_dict = pickle.load(pkl_i)
    #change label_processed by third party labels, threshold is the same as self sentiment, which is 4
    data_dict['label'] = {}
    data_dict['label_processed'] = {}
    for index in data_dict['label_thirdpartysentiment']:
        #some label is not well formatted, such as '6 delete', ' 4', so strip the space ' ' first, then choose the 0 index, this will choose the exact number
        data_dict['label'][index] = 'high' if round(data_dict['label_thirdpartysentiment'][index]) > 4 else 'low'
    for index in data_dict['label']:
        data_dict['label_processed'][index] = info_dict['label2id'][data_dict['label'][index]]
    
    with open(os.path.join(data_root, change_to_data_names[data_choose], data_file), 'wb') as pkl_o:
        pickle.dump(data_dict, pkl_o)

with open(os.path.join(data_root, change_to_data_names[data_choose], 'Pre_InfoDict.pkl'), 'wb') as pkl_o:
    pickle.dump(info_dict, pkl_o)