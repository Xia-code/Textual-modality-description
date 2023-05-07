# -*- coding: utf-8 -*-
"""
Created on Mon Jun  6 11:45:08 2022

@author: 30816
"""

import os
import argparse

config_arguments = argparse.ArgumentParser(description='Experiment configuration')

config_arguments.add_argument('-hidden_size', '--hidden_size', nargs='*', type=int, required=False, default=[256, 256], help='dnn middle size')
config_arguments.add_argument('-batch_size', '--batch_size', type=int, required=False, default=32, help='batch size')
config_arguments.add_argument('-epoch', '--epoch', type=int, required=False, default=200, help='training epoch')
config_arguments.add_argument('-folds', '--folds', type=int, required=False, default=5, help='folds')
config_arguments.add_argument('-fold_train', '--fold_train', action='store_true', help='training in folds or not; if False, then split the dataset by default setting: IEMOCAP sess1-4 train sess 5 test; Hazumi first 80% train, remained 20% test')
config_arguments.add_argument('-has_valid', '--has_valid', action='store_true', help='split validation set or not')
config_arguments.add_argument('-early_stop', '--early_stop', type=int, required=False, default=5, help='early stop')
config_arguments.add_argument('-ep_before_early_stop', '--ep_before_early_stop', type=int, required=False, default=5, help='epoches that not count for early stop in cold start')
config_arguments.add_argument('-lr', '--lr', type=float, required=False, default=0.001, help='learning rate')
config_arguments.add_argument('-train_percent', '--train_percent', type=float, required=False, default=0.9, help='train/valid split percent')
config_arguments.add_argument('-dropout', '--dropout', type=float, required=False, default=0.1, help='drop out')
config_arguments.add_argument('-ex_group', '--ex_group', type=str, required=False, default='FreezeModelBert', help='experiment group')
config_arguments.add_argument('-ex_dataset', '--ex_dataset', type=str, required=False, default='IEMOCAP', help='experiment dataset')
config_arguments.add_argument('-modality', '--modality', type=str, required=False, default='ALL', help='modality, ALL means loop all modalities; if set for specific modality, then only run the specific modality')
config_arguments.add_argument('-pooling', '--pooling', type=str, required=False, default='CLS', help='pooling method')
config_arguments.add_argument('-test_flag', '--test_flag', type=int, required=False, default=0, help='test flag')

config_vars = vars(config_arguments.parse_args())

class configuration():
    def __init__(self, 
                 input_size=400, 
                 modality_input_sizes={'L': 768, 'A': 384, 'F': 18}, 
                 lstm_layers=2, 
                 lstm_hidden_size=256, 
                 lstm_bidirectional=True, 
                 linear_n_layers=3, 
                 linear_middle_size=[256, 256, 256], 
                 label_size=18, 
                 batch_size=8, 
                 dropout=0.3, 
                 epoch=15, 
                 folds=5, 
                 early_stop=5, 
                 ep_before_early_stop=3, 
                 ex_group='Bert', 
                 ex_dataset='IEMOCAP', 
                 pretrained_pooling='CLS', 
                 model_type='LSTM', 
                 result_save_root='../results', 
                 result_save_dir='G1', 
                 modalities='LF', 
                 modalities_save_path='../results/G1/LF', 
                 load_model_flag=0, 
                 load_model_root='../results', 
                 load_model_path='../results/BestModel.pth', 
                 load_model_state_dict_path='../results/BestModelStateDict.pth', 
                 lr=0.001,
                 GPU_USE=True, 
                 random_seed=1, 
                 info_dict={}, 
                 fold_num=1, 
                 test_flag=0):
        self.input_size = input_size
        self.modality_input_sizes = modality_input_sizes
        self.lstm_layers = lstm_layers
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_bidirectional = lstm_bidirectional
        self.linear_middle_size = linear_middle_size
        self.model_type = model_type
        self.label_size = label_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.epoch = epoch
        self.folds = folds
        self.early_stop = early_stop
        self.ep_before_early_stop = ep_before_early_stop
        self.ex_group = ex_group
        self.ex_dataset = ex_dataset
        self.pretrained_pooling = pretrained_pooling
        self.result_save_root = result_save_root
        self.result_save_dir = result_save_dir
        self.load_model_flag = load_model_flag
        self.load_model_root = load_model_root
        self.load_model_path = load_model_path
        self.load_model_state_dict_path = load_model_state_dict_path
        self.modalities = modalities
        self.lr = lr
        self.fold_num = fold_num
        self.GPU_USE = GPU_USE
        self.random_seed = random_seed
        self.info_dict = info_dict
        self.test_flag = test_flag
        if(self.lstm_bidirectional):
            self.lstm_output_size = self.lstm_hidden_size * 2
        else:
            self.lstm_output_size = self.lstm_hidden_size
        
        self.modalities_save_path = modalities_save_path
        self.logging_save_path = os.path.join(modalities_save_path, 'log_f' + str(fold_num) + '.log')
        self.random_seed_save_path = os.path.join(modalities_save_path, 'random_seed.txt')
        self.modalities_save_path = self.modalities_save_path
        self.model_save_path = os.path.join(self.modalities_save_path, 'BestModel.pth')
        self.model_state_dict_save_path = os.path.join(self.modalities_save_path, 'BestModelStateDict.pth')
        self.result_save_path = os.path.join(self.modalities_save_path, 'Results_f' + str(fold_num) + '.pkl')
        self.metric_result_save_path = os.path.join(self.modalities_save_path, 'Results_metrics_f' + str(fold_num) + '.txt')
        self.all_fold_result_save_path = os.path.join(self.modalities_save_path, 'AllFoldResults.pkl')
        self.aver_metric_result_save_path = os.path.join(self.modalities_save_path, 'AverResults_metrics.txt')