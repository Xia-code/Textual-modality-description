# -*- coding: utf-8 -*-
"""
This script is for summarizing results from different models and writing out
"""

import os
import csv
import glob
import pickle
import numpy as np
from sklearn.metrics import confusion_matrix

result_root = '../results_'
result_save_root = '../result_sta'
dataset = 'HazumiThird' #Hazumi, HazumiThird

pooling_list = ['CLS', 'MEAN']
model_list = ['FreezeBert', 'FreezeRoBerta', 'Bert', 'RoBerta', 'DNN', 'DNNEARLY', 'DNNLATE1', 'DNNLATE2']
metric_list = ['UA', 'WA', 'macro_F1', 'CM']

all_results = glob.glob(os.path.join(result_root, '*_' + dataset + '_*'))

write_dict = {}
modality_list = ['A', 'F', 'A+F', 'L', 'L+A', 'L+F', 'L+A+F']
#init write metrics
for key in metric_list:
    write_dict[key] = {}
    #init pooling keys
    for pooling in pooling_list:
        write_dict[key][pooling] = {}
        #get modalities and init modality keys
        for result in all_results:
            result_dir = result.split('\\')[-1]
            result_split = result_dir.split('_')
            modality_dirs = glob.glob(os.path.join(result, '*'))
            for modality in modality_list:
                write_dict[key][pooling][modality] = {}
            break

for pooling in pooling_list:
    for model_key in model_list:
        temp_results = glob.glob(os.path.join(result_root, '*_' + model_key + '_' + dataset + '_*'))
        for result_dir in temp_results:
            result_name = result_dir.split('\\')[-1]
            result_split = result_name.split('_')
            ex_group = result_split[1]
            ex_dataset = result_split[2] #just for confirmation, dataset was bounded in glob at the begining
            if(ex_dataset != dataset):
                print(result_dir)
            
            continue_flag = 0
            pooling_method = result_split[5] if (model_key[:3] != 'DNN') else 'PoolingCLS'
            if(pooling_method != 'Pooling' + pooling):
                continue_flag = 1
            
            if(continue_flag == 1):
                continue
            else:
                #read acc (UA) and macro F1 from txt file
                #read CM from pkl file
                for modality in modality_list:
                    modality_dir = os.path.join(result_dir, modality)
                    if(os.path.exists(os.path.join(modality_dir, 'AverResults_metrics.txt')) == False or
                       os.path.exists(os.path.join(modality_dir, 'AllFoldResults.pkl')) == False):
                        write_dict['UA'][pooling][modality][result_name] = -10000
                        write_dict['macro_F1'][pooling][modality][result_name] = -10000 #define a missing number that is obvious issued
                        write_dict['CM'][pooling][modality][result_name] = np.zeros((1, 1))
                        write_dict['WA'][pooling][modality][result_name] = -10000
                    else:
                        with open(os.path.join(modality_dir, 'AverResults_metrics.txt'), 'r') as f_in:
                            line = f_in.readline().strip('\n')
                        write_dict['UA'][pooling][modality][result_name] = float(line.split(' | ')[0].split(' : ')[1])
                        write_dict['macro_F1'][pooling][modality][result_name] = float(line.split(' | ')[2].split(' : ')[1])
                        with open(os.path.join(modality_dir, 'AllFoldResults.pkl'), 'rb') as pkl_i:
                            foldresults = pickle.load(pkl_i)
                        #compute average confusion matrix and WA
                        count = 0
                        for fold in foldresults:
                            if(type(fold) != int):
                                continue
                            else:
                                if(count == 0):
                                    fold_cm = confusion_matrix(foldresults[fold]['test_true_list'], 
                                                               foldresults[fold]['test_pred_list'])
                                else:
                                    fold_cm += confusion_matrix(foldresults[fold]['test_true_list'], 
                                                                foldresults[fold]['test_pred_list'])
                                count += 1
                        aver_cm = fold_cm / count
                        write_dict['CM'][pooling][modality][result_name] = aver_cm
                        label_acc = {}
                        for cm_i in range(aver_cm.shape[0]):
                            label_acc[cm_i] = aver_cm[cm_i][cm_i] / sum(aver_cm[cm_i])
                        write_dict['WA'][pooling][modality][result_name] = sum([label_acc[l] for l in label_acc]) / len(label_acc)
#output write_dict, because CM is not easy to write out as csv file, so save write dict for investigating CM later
with open(os.path.join(result_save_root, result_root.split('/')[-1] + '.pkl'), 'wb') as pkl_o:
    pickle.dump(write_dict, pkl_o)

#output csv file divided by metric
for metric in metric_list:
    if(metric != 'CM'):
        write_str_list = []
        for pooling in pooling_list:
            first_row = [pooling] + modality_list
            write_str_list.append(first_row)
            for modality in modality_list:
                ex_list = (write_dict[metric][pooling][modality].keys())
                break
            for ex in ex_list:
                ex_row = [pooling + '_' + ex]
                for modality in first_row[1:]:
                    ex_row.append(write_dict[metric][pooling][modality][ex])
                write_str_list.append(ex_row)
        #write csv by transposing write str list
        with open(os.path.join(result_save_root, dataset + '_' + metric + '.csv'), 'w', newline='') as csv_o:
            csv_writer = csv.writer(csv_o)
            for w_i in range(len(write_str_list[0])):
                output_list = []
                for w_j in range(len(write_str_list)):
                    output_list.append(write_str_list[w_j][w_i])
                csv_writer.writerow(output_list)