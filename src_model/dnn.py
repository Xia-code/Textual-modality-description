# -*- coding: utf-8 -*-
"""
This script is to define DNN models, including DNN-base, Early Fusion, Late Fusion 1, Late Fusion 2
"""

import torch
import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, config):
        super(DNN, self).__init__()
        self.input_linear_layer = nn.Linear(config.input_size, config.linear_middle_size[0])
        self.linear_middle_layers = nn.ModuleList([nn.Linear(config.linear_middle_size[i], config.linear_middle_size[i + 1]) for i in range(len(config.linear_middle_size) - 1)])
        self.mapping_layer = nn.Linear(config.linear_middle_size[-1], config.label_size)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        
    def forward(self, indata):
        output = self.dropout_layer(torch.tanh(self.input_linear_layer(indata)))
        for i in range(len(self.linear_middle_layers)):
            output = self.dropout_layer(torch.relu(self.linear_middle_layers[i](output)))
        output = self.mapping_layer(output)
        return output

class tanh_DNN(nn.Module):
    def __init__(self, config):
        super(tanh_DNN, self).__init__()
        self.input_linear_layer = nn.Linear(config.input_size, config.linear_middle_size[0])
        self.linear_middle_layers = nn.ModuleList([nn.Linear(config.linear_middle_size[i], config.linear_middle_size[i + 1]) for i in range(len(config.linear_middle_size) - 1)])
        self.mapping_layer = nn.Linear(config.linear_middle_size[-1], config.label_size)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        
    def forward(self, indata):
        output = self.dropout_layer(torch.tanh(self.input_linear_layer(indata)))
        for i in range(len(self.linear_middle_layers)):
            output = self.dropout_layer(torch.tanh(self.linear_middle_layers[i](output)))
        output = self.mapping_layer(output)
        return output

class DNN_early_fusion_1(nn.Module):
    def __init__(self, config):
        super(DNN_early_fusion_1, self).__init__()
        self.input_linear_layer = nn.Linear(config.input_size, 128)
        self.linear_middle_layer = nn.Linear(128, 128)
        self.later_middle_layer_1 = nn.Linear(128, 64)
        self.later_middle_layer_2 = nn.Linear(64, 64)
        self.mapping_layer = nn.Linear(64, config.label_size)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        
    def forward(self, indata):
        output = self.dropout_layer(torch.relu(self.input_linear_layer(indata)))
        output = self.dropout_layer(torch.relu(self.linear_middle_layer(output)))
        output = self.dropout_layer(torch.relu(self.later_middle_layer_1(output)))
        output = self.dropout_layer(torch.relu(self.later_middle_layer_2(output)))
        output = self.mapping_layer(output)
        return output

class DNN_late_fusion_1(nn.Module):
    def __init__(self, config):
        super(DNN_late_fusion_1, self).__init__()
        self.modality2id = {'L': 0, 'A': 1, 'F':2}
        self.input_linear_layer = nn.ModuleDict({})
        self.linear_middle_layers = nn.ModuleDict({})
        for m_d in config.modalities.split('+'):
            self.input_linear_layer[m_d] = nn.Linear(config.modality_input_sizes[m_d], 64)
            self.linear_middle_layers[m_d] = nn.Linear(64, 64)
        self.later_middle_layer_1 = nn.Linear(64 * len(config.modalities.split('+')), 32)
        self.later_middle_layer_2 = nn.Linear(32, 32)
        self.mapping_layer = nn.Linear(32, config.label_size)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        
    def forward(self, indata):
        output_list = []
        for key in indata:
            early_output = self.dropout_layer(torch.relu(self.input_linear_layer[key](indata[key])))
            early_output = self.dropout_layer(torch.relu(self.linear_middle_layers[key](early_output)))
            output_list.append(early_output)
        output = torch.cat(output_list, dim=-1)
        output = self.dropout_layer(torch.relu(self.later_middle_layer_1(output)))
        output = self.dropout_layer(torch.relu(self.later_middle_layer_2(output)))
        output = self.mapping_layer(output)
        return output

class DNN_late_fusion_2(nn.Module):
    def __init__(self, config):
        super(DNN_late_fusion_2, self).__init__()
        self.modality2id = {'L': 0, 'A': 1, 'F':2}
        self.input_linear_layer = nn.ModuleDict({})
        self.linear_middle_layers = nn.ModuleDict({})
        self.later_middle_layer_1 = nn.ModuleDict({})
        self.later_middle_layer_2 = nn.ModuleDict({})
        for m_d in config.modalities.split('+'):
            self.input_linear_layer[m_d] = nn.Linear(config.modality_input_sizes[m_d], 64)
            self.linear_middle_layers[m_d] = nn.Linear(64, 64)
            self.later_middle_layer_1[m_d] = nn.Linear(64, 32)
            self.later_middle_layer_2[m_d] = nn.Linear(32, 32)
        self.mapping_layer = nn.Linear(32, config.label_size)
        self.dropout_layer = nn.Dropout(p=config.dropout)
        
    def forward(self, indata):
        output_list = []
        for key in indata:
            output = self.dropout_layer(torch.relu(self.input_linear_layer[key](indata[key])))
            output = self.dropout_layer(torch.relu(self.linear_middle_layers[key](output)))
            output = self.dropout_layer(torch.relu(self.later_middle_layer_1[key](output)))
            output = self.dropout_layer(torch.relu(self.later_middle_layer_2[key](output)))
            output = self.mapping_layer(output)
            output_list.append(output)
        for ii in range(len(output_list)):
            if(ii == 0):
                return_output = output_list[ii]
            else:
                return_output += output_list[ii]
        return return_output