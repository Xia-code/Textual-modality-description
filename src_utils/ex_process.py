# -*- coding: utf-8 -*-
"""
Created on Wed Jun  1 14:27:15 2022

@author: 30816
"""

import numpy as np
import torch
import torch.utils.data
from torch.utils.data import DataLoader
from sklearn.utils.class_weight import compute_class_weight

def split_fold_data_no_valid(data_dict, info_dict, config_vars, model_key, modality_key, fold, fold_train_group=None):
    data_split_dict = {'train': [], 'test': []}
    if(config_vars['ex_dataset'].split('-')[0] == 'IEMOCAP'):
        for index in data_dict['file_name']:
            if(int(data_dict['file_name'][index][4]) == fold + 1):
                data_split_dict['test'].append(index)
            else:
                data_split_dict['train'].append(index)
    elif(config_vars['ex_dataset'].split('-')[0] == 'Hazumi'):
        for index in data_dict['group']:
            if(data_dict['group'][index] not in fold_train_group[fold + 1]):
                data_split_dict['test'].append(index)
            else:
                data_split_dict['train'].append(index)
    data_split_keys = ['train', 'test']
    text_list_dict = {}
    label_list_dict = {}
    for key in data_split_keys:
        text_list_dict[key] = []
        label_list_dict[key] = []
        for index in data_split_dict[key]:
            if(config_vars['ex_group'][:6] == 'Freeze'): #for freeze model experiment, only use the data from the set pooling group
                text_list_dict[key].append(data_dict['freeze_model_repr'][model_key][config_vars['pooling']][modality_key][index])
                label_list_dict[key].append(data_dict['label_processed'][index])
            else:
                text_list_dict[key].append(data_dict['modality_text'][model_key][modality_key][index])
                label_list_dict[key].append(data_dict['label_processed'][index])
    
    if(config_vars['ex_group'][:6] == 'Freeze'):
        for key in data_split_keys:
            text_list_dict[key] = torch.from_numpy(np.concatenate(text_list_dict[key], axis=0)).float()
            label_list_dict[key] = torch.from_numpy(np.array(label_list_dict[key])).long()
    
    train_dataset = data_set(text_list_dict['train'], label_list_dict['train'])
    test_dataset = data_set(text_list_dict['test'], label_list_dict['test'])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config_vars['batch_size'], shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    label_weight_tensor = get_label_weights(label_list_dict['train'], info_dict['id2label'], sequence_flag=0)
    return train_dataloader, test_dataloader, label_weight_tensor

def split_fold_data_has_valid(data_dict, info_dict, config_vars, model_key, modality_key, fold, fold_train_group=None):
    data_split_dict = {'train': [], 'valid': [], 'test': []}
    train_index_temp = []
    if(config_vars['ex_dataset'][:7] == 'IEMOCAP'):
        for index in data_dict['file_name']:
            if(int(data_dict['file_name'][index][4]) == fold +1):
                data_split_dict['test'].append(index)
            else:
                train_index_temp.append(index)
    elif(config_vars['ex_dataset'][:6] == 'Hazumi'):
        for index in data_dict['group']:
            if(data_dict['group'][index] not in fold_train_group[fold + 1]):
                data_split_dict['test'].append(index)
            else:
                train_index_temp.append(index)
    
    train_fold_index_temp = {} #split training and validation set by TRAIN_PERCENT from training data
    for index in train_index_temp:
        if(data_dict['label_processed'][index] not in train_fold_index_temp.keys()):
            train_fold_index_temp[data_dict['label_processed'][index]] = [index]
        else:
            train_fold_index_temp[data_dict['label_processed'][index]].append(index)
    for key in train_fold_index_temp:
        valid_split = int(len(train_fold_index_temp[key]) * config_vars['train_percent'])
        data_split_dict['train'] += train_fold_index_temp[key][:valid_split]
        data_split_dict['valid'] += train_fold_index_temp[key][valid_split:]
    
    data_split_keys = ['train', 'valid', 'test']
    text_list_dict = {}
    label_list_dict = {}
    for key in data_split_keys:
        text_list_dict[key] = []
        label_list_dict[key] = []
        for index in data_split_dict[key]:
            if(config_vars['ex_group'][:6] == 'Freeze'):
                text_list_dict[key].append(data_dict['freeze_model_repr'][model_key][config_vars['pooling']][modality_key][index])
                label_list_dict[key].append(data_dict['label_processed'][index])
            else:
                text_list_dict[key].append(data_dict['modality_text'][model_key][modality_key][index])
                label_list_dict[key].append(data_dict['label_processed'][index])
    
    if(config_vars['ex_group'][:6] == 'Freeze'):
        for key in data_split_keys:
            text_list_dict[key] = torch.from_numpy(np.concatenate(text_list_dict[key], axis=0)).float()
            label_list_dict[key] = torch.from_numpy(np.array(label_list_dict[key])).long()
    
    train_dataset = data_set(text_list_dict['train'], label_list_dict['train'])
    valid_dataset = data_set(text_list_dict['valid'], label_list_dict['valid'])
    test_dataset = data_set(text_list_dict['test'], label_list_dict['test'])
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config_vars['batch_size'], shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=1, shuffle=False)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    label_weight_tensor = get_label_weights(label_list_dict['train'], info_dict['id2label'], sequence_flag=0)
    return train_dataloader, valid_dataloader, test_dataloader, label_weight_tensor

class data_set(torch.utils.data.Dataset):
    def __init__(self, data_dict, label_dict):
        self.data_dict = data_dict
        self.label_dict = label_dict
    
    def __getitem__(self, index):
        return self.data_dict[index], self.label_dict[index]
    
    def __len__(self):
        return len(self.data_dict)

class multitask_data_set(torch.utils.data.Dataset):
    def __init__(self, data_tensor, label1_tensor, label2_tensor):
        self.data_tensor = data_tensor
        self.label1_tensor = label1_tensor
        self.label2_tensor = label2_tensor
    
    def __getitem__(self, index):
        return self.data_tensor[index], self.label1_tensor[index], self.label2_tensor[index]
    
    def __len__(self):
        return len(self.data_tensor)

def dataloader_collate_fn_sequence(data):
    input_data, label = zip(*data)
    lengths = [len(bs_x) for bs_x in input_data]
    max_lengths = max(lengths)
    padded_seqs = torch.Tensor(len(input_data), max_lengths).fill_(0).float()
    for i, seq in enumerate(input_data):
        length = lengths[i]
        padded_seqs[i, :length] = torch.Tensor(seq).float()
    return padded_seqs, label

def get_label_weights(label_data, id2label, sequence_flag=0):
    label_flatten = []
    if(sequence_flag == 0):
        label_flatten = np.array(label_data)
    if(sequence_flag == 1):
        label_flatten = []
        for i in range(label_data.shape[0]):
            for j in range(label_data.shape[1]):
                if(label_data[i][j] != -1):
                    label_flatten.append(label_data[i][j])
    label_weight = compute_class_weight('balanced', classes=np.unique(label_flatten), y=label_flatten)
    label_weight_tensor = torch.from_numpy(label_weight).float()
    return label_weight_tensor

def manual_get_label_weights(label_data, id2label, sequence_flag=0):
    label_count_dict = {}
    for l in id2label:
        label_count_dict[l] = 0
    if(sequence_flag == 0):
        for l in label_data:
            if(l != -1):
                label_count_dict[l] += 1
    if(sequence_flag == 1):
        for label_list in label_data:
            for l in label_list:
                if(l != -1):
                    label_count_dict[l] += 1
    label_weight_tensor = torch.zeros(len(label_count_dict.keys()))
    for l in range(len(label_count_dict.keys())):
        if(l != -1):
            if(label_count_dict[l] == 0):
                label_weight_tensor[l] = 1
            else:
                label_weight_tensor[l] = 1 / label_count_dict[l]
    print(label_weight_tensor)
    return label_weight_tensor

def get_experiment_samples(ex_data, dataset_split={'train': [1, 2, 3, 4, 5, 6, 7, 8], 'valid': [9], 'test': [10, 11]}, sequence_flag=0, sequence_len=10, shift=20):
    sample_dict = {}
    sample_dict['train'] = []
    sample_dict['valid'] = []
    sample_dict['test'] = []
    for_index_data = ex_data['Input_text_index']
    for file in for_index_data:
        file_group = int(file.split('-')[0]) if file[0] != 'A' else int(file.split('-')[0][1:])
        index_list = list(for_index_data[file].keys())
        if(sequence_len > len(index_list)):
            raise(ValueError('Sequence length too large, should lower than the fold interval'))
        for key in dataset_split:
            if(file_group in dataset_split[key]):
                append_key = key
        if(sequence_flag == 0):
            for j in index_list:
                sample_dict[append_key].append((file, j))
        elif(sequence_flag == 1):
            if(append_key == 'train'):
                for j in range(int(len(index_list) / shift)):
                    sample_dict[append_key].append((file, index_list[j * shift: j * shift + sequence_len]))
            else:
                if(len(index_list) % sequence_len != 0):
                    iter_n = int(len(index_list) / sequence_len) + 1
                else:
                    iter_n = int(len(index_list) / sequence_len)
                for j in range(iter_n):
                    sample_dict[append_key].append((file, index_list[j * sequence_len: j * sequence_len + sequence_len]))
    ex_data['sample_dict'] = sample_dict
    ex_data['sequence_len'] = sequence_len
    return ex_data
    
class log_writer():
    def __init__ (self, filename):
        self.filename = filename
    
    def write(self, log_contents, write_flag='a'):
        with open(self.filename, write_flag) as log_o:
            print(log_contents)
            log_o.write(log_contents)
            log_o.write('\n')