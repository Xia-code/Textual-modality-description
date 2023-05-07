# -*- coding: utf-8 -*-
"""
This scrip is the training process for Late Fusion 1 and 2 with validations
"""

import os
import sys
sys.path.append('../src_10x_utils')
sys.path.append('../src_10x_model')
import time
import tqdm
import pickle
import random
import numpy as np
import dnn
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim

from ex_process import log_writer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

def train_process(train_dataloader, valid_dataloader, test_dataloader, label_weights, model, ex_info_dict, 
                  configs, sequence_flag, GPU_USE):
    
    best_weighted_f1 = 0
    best_acc = 0
    best_f1 = 0
    early_stop_count = 0
    break_flag = 0
    
    result_dict = {}
    for key in ['acc', 'weighted_f1', 'macro_f1', 'test_acc', 'test_weighted_f1', 'test_macro_f1']:
        result_dict[key] = 0
    for key in ['pred_list', 'true_list', 'test_pred_list', 'test_true_list']:
        result_dict[key] = []
    
    with open(configs.random_seed_save_path, 'w') as f_o:
        f_o.write(str(configs.random_seed))
    logger = log_writer(configs.logging_save_path)
    logger.write('Start Fold ' + str(configs.fold_num) + ' Training...')
    for ep in range(configs.epoch):
        if(ep <= 10):
            lrr = 0.001
        elif(ep <= 50):
            lrr = 0.001
        else:
            lrr = 0.0005
        optimizer = optim.Adam(params=model.parameters(), lr=lrr)
        pbar_train = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), position=0)
        pbar_valid = tqdm.tqdm(enumerate(valid_dataloader), total=len(valid_dataloader), position=0)
        criteria = nn.CrossEntropyLoss(weight=label_weights, ignore_index=-1)
        criteria = nn.CrossEntropyLoss(ignore_index=-1)
        acc, weighted_f1, macro_f1, pred_list, true_list = train_step(model, pbar_train, pbar_valid, criteria, 
                                                optimizer, configs, logger, ep, sequence_flag=sequence_flag, 
                                                use_label=configs.ex_dataset.lower())
        if(acc > best_acc):
            best_weighted_f1 = weighted_f1
            best_acc = acc
            best_f1 = macro_f1
            torch.save(model, configs.model_save_path)
            torch.save(model.state_dict(), configs.model_state_dict_save_path)
            result_dict['pred_list'] = pred_list
            result_dict['true_list'] = true_list
            result_dict['acc'] = acc
            result_dict['weighted_f1'] = weighted_f1
            result_dict['macro_f1'] = macro_f1
            
            early_stop_count = 0
            logger.write('Better Model Found!')
            logger.write('Fold: ' + str(configs.fold_num) + ' | ' + configs.modalities + ' | Epoch: ' + str(ep + 1) + '\nAcc: ' + str(round(acc, 5))
                      + ' | Weighted_F1: ' + str(round(weighted_f1, 5)) + ' | Macro_F1: ' + str(round(macro_f1, 5)))
        elif(ep > configs.ep_before_early_stop):
            if(early_stop_count < configs.early_stop):
                early_stop_count += 1
                logger.write('No Better Model Found (%d / %d)' %(early_stop_count, configs.early_stop))
                logger.write(configs.modalities + ' | Epoch: ' + str(ep + 1) + '\nAcc: ' + str(round(acc, 5))
                      + ' | Weighted_F1: ' + str(round(weighted_f1, 5)) + ' | Macro_F1: ' + str(round(macro_f1, 5)))
            elif(early_stop_count >= configs.early_stop):
                logger.write('No Better Model Found (%d / %d), training finished' %(early_stop_count, configs.early_stop))
                logger.write(configs.modalities + ' | Epoch: ' + str(ep + 1) + '\nAcc: ' + str(round(acc, 5))
                      + ' | Weighted_F1: ' + str(round(weighted_f1, 5)) + ' | Macro_F1: ' + str(round(macro_f1, 5)))
                break_flag = 1
        else:
            logger.write(configs.modalities + ' Epoch: ' + str(ep + 1) + '\nAcc: ' + str(round(acc, 5))
                      + ' | Weighted_F1: ' + str(round(weighted_f1, 5)) + ' | Macro_F1: ' + str(round(macro_f1, 5)))
        if(break_flag == 1):
            logger.write(configs.modalities + ' OVER\n')
            break
        if(configs.test_flag == 1):
            torch.save(model, configs.model_save_path)
            torch.save(model.state_dict(), configs.model_state_dict_save_path)
            break
    with open(configs.metric_result_save_path, 'w') as f_o:
        f_o.write('Valid results: Acc : ' + str(round(best_acc, 5)) + ' | Weighted F1 : ' + str(round(best_weighted_f1, 5)) + ' | Macro F1 : ' + str(round(best_f1, 5)))
    # test process
    logger.write('Start Test Process')
    test_model = torch.load(configs.model_save_path)
    test_model.load_state_dict(torch.load(configs.model_state_dict_save_path))
    pbar_test = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader), position=0)
    test_acc, test_weighted_f1, test_macro_f1, test_pred_list, test_true_list = valid_test_process(test_model, pbar_test, configs, sequence_flag=sequence_flag, use_label=configs.ex_dataset.lower())
    result_dict['test_acc'] = test_acc
    result_dict['test_weighted_f1'] = test_weighted_f1
    result_dict['test_macro_f1'] = test_macro_f1
    result_dict['test_pred_list'] = test_pred_list
    result_dict['test_true_list'] = test_true_list
    logger.write(configs.modalities + ' Test Results: \nAcc: ' + str(round(test_acc, 5))
                      + ' | Weighted_F1: ' + str(round(test_weighted_f1, 5)) + ' | Macro_F1: ' + str(round(test_macro_f1, 5)))
    with open(configs.result_save_path, 'wb') as pkl_o:
        pickle.dump(result_dict, pkl_o)
    with open(configs.metric_result_save_path, 'a') as f_o:
        f_o.write('\n')
        f_o.write('Test results: Acc : ' + str(round(test_acc, 5)) + ' | Weighted F1 : ' + str(round(test_weighted_f1, 5)) + ' | Macro F1 : ' + str(round(test_macro_f1, 5)))
    os.remove(configs.model_save_path)
    os.remove(configs.model_state_dict_save_path)
    return result_dict
    
def train_step(model, pbar_train, pbar_valid, criteria, optimizer, configs, logger, ep, sequence_flag=0, use_label='da'):
    model.train()
    loss_list = []
    ccc = 0
    for i, (x_data, label) in pbar_train:
        if(configs.GPU_USE):
            for key in x_data:
                x_data[key] = x_data[key].cuda()
            label = label.cuda()
        optimizer.zero_grad()
        if(configs.model_type != 'LSTM_CRF'):
            model_output = model(x_data)
            if(sequence_flag == 0):
                loss = criteria(model_output, label)
            if(sequence_flag == 1):
                for j, _ in enumerate(model_output):
                    if(j == 0):
                        loss = criteria(model_output[j], label[j])
                    else:
                        loss = loss + criteria(model_output[j], label[j])
                loss = loss / configs.batch_size
        elif(configs.model_type == 'LSTM_CRF'):
            mask = torch.ge(label, 0)
            if(configs.GPU_USE):
                mask = mask.cuda()
            model_output = model(x_data, label, mask)
            loss = model_output
        
        loss.backward()
        optimizer.step()
        loss_list.append(loss.item())
        torch.cuda.empty_cache()
        #pbar_train.set_description("(Epoch {}) LOSS : {:.4f}".format((ep + 1), loss.item()))
        ccc += 1
        if(configs.test_flag == 1):
            if(ccc > 10):
                break
        #if(i % 200 == 0):
    logger.write('Fold: ' + str(configs.fold_num) + ' Epoch %d Average Loss : %.5f' %(ep + 1, (sum(loss_list) / len(loss_list))))
    acc, weighted_f1, macro_f1, pred_list, true_list = valid_test_process(model, pbar_valid, configs, sequence_flag=sequence_flag, use_label=use_label)
    return acc, weighted_f1, macro_f1, pred_list, true_list

def valid_test_process(model, pbar_data, configs, sequence_flag=0, use_label='da', mode='valid'):
    model.eval()
    pred_list = []
    true_list = []
    for i, (x_data, label) in pbar_data:
        if(configs.GPU_USE):
            for key in x_data:
                x_data[key] = x_data[key].cuda()
        if(configs.model_type != 'LSTM_CRF'):
            model_output = torch.softmax(model(x_data), dim=-1)
            #print(x_data)
            #print(model_output)
            if(sequence_flag == 0):
                if(configs.GPU_USE):
                    output_labels = list(torch.argmax(model_output, dim=-1).cpu().numpy())
                else:
                    output_labels = list(torch.argmax(model_output, dim=-1).numpy())
                pred_list += output_labels
                true_list += list(label.numpy())
            if(sequence_flag == 1):
                if(configs.GPU_USE):
                    output_labels = list(torch.argmax(torch.softmax(model_output[0], dim=-1), -1).cpu().numpy())
                else:
                    output_labels = list(torch.argmax(torch.softmax(model_output[0], dim=-1), -1).numpy())
                for l_i, l_t in enumerate(list(label[0].numpy())):
                    if(l_t != -1):
                        pred_list.append(output_labels[l_i])
                        true_list.append(l_t)
        elif(configs.model_type == 'LSTM_CRF'):
            mask = torch.ge(label, 0)
            if(configs.GPU_USE):
                mask = mask.cuda()
            model_output = model.decode(x_data, mask)
            output_labels = model_output
            for l_i, l_t in enumerate(list(label[0].numpy())):
                if(l_t != -1):
                    pred_list.append(output_labels[l_i])
                    true_list.append(l_t)
        if(configs.test_flag == 1):
            if(i > 10):
                break
    acc = accuracy_score(true_list, pred_list)
    weighted_f1 = f1_score(true_list, pred_list, average='weighted')
    macro_f1 = f1_score(true_list, pred_list, average='macro')
    return acc, weighted_f1, macro_f1, pred_list, true_list