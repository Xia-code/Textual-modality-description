# -*- coding: utf-8 -*-
"""
This script is to clean the data to discard participant with missing audio, utterances with missing labels
And extract necessary data from original dataset and modalities
In extracting, we rely on alignment of participant number-time order (time stamp)-start and end time-utterance index, to make data from different modalities and annotations correctly aligned

Necessary data include start-end time (for segmenting audio), action units obtained from open face (this is obtained from previous study, but with utterance index, so we can align AUs with other processed data)
"""

import os
import glob
import csv
import pickle
import xml.dom.minidom
import numpy as np

annotation_root = '../Data_ori/Hazumi/SelfSentimentAnnotation' #from annotations read start/end time and AUs
thirdparty_annotation_root = '../Data_ori/Hazumi/ThirdPartySentimentAnnotation'
transcript_root = '../Data_ori/Hazumi/Transcript' #from transcript read text
synchro_root = '../Data_ori/Hazumi/TimeSynchro' #get precise start/end time of user
au_root = '../Data_ori/Hazumi/SegData'
save_root = '../Data/Hazumi'

if(os.path.exists(os.path.join(save_root)) == False):
    os.mkdir(save_root)

data_dict = {}

AU_list = ['AU01_c', 'AU02_c', 'AU04_c', 'AU05_c', 'AU06_c', 'AU07_c', 
           'AU09_c', 'AU10_c', 'AU12_c', 'AU14_c', 'AU15_c', 'AU17_c', 
           'AU20_c', 'AU23_c', 'AU25_c', 'AU26_c', 'AU28_c', 'AU45_c']

info_dict = {}
info_dict['AU_list'] = AU_list

for group in ['1902', '1911']:
    transcripts = glob.glob(os.path.join(transcript_root, group, '*.eaf'))
    for transcrip in transcripts:
        speaker = transcrip.split('\\')[-1].split('.csv')[0].split('_')[0]
        
        #init data_dict for the speaker key
        data_dict[speaker] = {}
        for key in ['text', 'start_time_user', 'end_time_user', 'start_time_agent', 'AU', 'label_selfsentiment', 'speaker', 'label_thirdpartysentiment', 
                    'has_label', #some files missed parts of video, so these parts were not annotated. this key record whether the corresponding index was annotated or not, for subsequential processes like audio segmentions
                    'has_all_thirdparty_label', 
                    'has_AU']: 
            data_dict[speaker][key] = {}
        
        #read transcript contents
        transcript_dom = xml.dom.minidom.parse(os.path.join(transcript_root, group, speaker + '_elan.eaf'))
        transcript_collection = transcript_dom.documentElement
        
        #read all time slots from transcript file first
        transcript_time_slot2value = {}
        time_slots = transcript_collection.getElementsByTagName("TIME_ORDER")[0].getElementsByTagName('TIME_SLOT')
        for time_slot in time_slots:
            transcript_time_slot2value[time_slot.getAttribute('TIME_SLOT_ID')] = time_slot.getAttribute('TIME_VALUE')
        
        #read annotation contents to assert whether time slots in annotations are the same in transcript
        annotation_dom = xml.dom.minidom.parse(os.path.join(annotation_root, group, speaker + '_elan.eaf'))
        annotation_collection = annotation_dom.documentElement
        
        #read all time slots from transcript file first
        annotation_time_slot2value = {}
        time_slots = annotation_collection.getElementsByTagName("TIME_ORDER")[0].getElementsByTagName('TIME_SLOT')
        for time_slot in time_slots:
            annotation_time_slot2value[time_slot.getAttribute('TIME_SLOT_ID')] = time_slot.getAttribute('TIME_VALUE')
        
        #read third party annotation contents
        thirdparty_collection_dict = {}
        thirdparty_time_slot2value_dict = {}
        thirdparty_annotation_paths = glob.glob(os.path.join(thirdparty_annotation_root, group, '*'))
        for thirdparty_annotation_path in thirdparty_annotation_paths:
            annotator = thirdparty_annotation_path.split('\\')[-1]
            thirdparty_annotation_dom = xml.dom.minidom.parse(os.path.join(thirdparty_annotation_path, speaker + '_elan2.eaf'))
            thirdparty_collection_dict[annotator] = thirdparty_annotation_dom.documentElement
            thirdparty_time_slot2value_dict[annotator] = {}
            time_slots = thirdparty_collection_dict[annotator].getElementsByTagName("TIME_ORDER")[0].getElementsByTagName('TIME_SLOT')
            for time_slot in time_slots:
                thirdparty_time_slot2value_dict[annotator][time_slot.getAttribute('TIME_SLOT_ID')] = time_slot.getAttribute('TIME_VALUE')
        
        #confirm whether time slot info are the same
        if(len(transcript_time_slot2value) != len(annotation_time_slot2value)):
            print(speaker)
        for annotator in thirdparty_time_slot2value_dict:
            if(len(transcript_time_slot2value) != len(thirdparty_time_slot2value_dict[annotator])):
                print(speaker)
        for time_slot_id in transcript_time_slot2value:
            if(abs(int(transcript_time_slot2value[time_slot_id]) - int(annotation_time_slot2value[time_slot_id])) > 10):
                #sometimes the annotations may have 1 ms difference, so confirm whether the difference is bigger than 10 for redundancy
                print(speaker, time_slot_id)
                
        transcript_contents = transcript_collection.getElementsByTagName('ALIGNABLE_ANNOTATION')
        annotation_contents = annotation_collection.getElementsByTagName('ALIGNABLE_ANNOTATION')
        thirdparty_contents_dict = {}
        for annotator in thirdparty_collection_dict:
            thirdparty_contents_dict[annotator] = thirdparty_collection_dict[annotator].getElementsByTagName('ALIGNABLE_ANNOTATION')
        
        for index in range(len(transcript_contents)): #after confimation, the length and indices of transcipt and annotation should be the same, so use the same index to get contents
            if (len(transcript_contents[index].getElementsByTagName('ANNOTATION_VALUE')[0].childNodes)==0):
                text=''
            else:
                text = transcript_contents[index].getElementsByTagName('ANNOTATION_VALUE')[0].childNodes[0].data
            if (len(annotation_contents[index].getElementsByTagName('ANNOTATION_VALUE')[0].childNodes)==0):
                label=''
            else:
                label = annotation_contents[index].getElementsByTagName('ANNOTATION_VALUE')[0].childNodes[0].data
                
            thirdparty_label_list = []
            for annotator in thirdparty_contents_dict:
                if (len(thirdparty_contents_dict[annotator][index].getElementsByTagName('ANNOTATION_VALUE')[0].childNodes)==0):
                    thirdparty_label=''
                else:
                    thirdparty_label = thirdparty_contents_dict[annotator][index].getElementsByTagName('ANNOTATION_VALUE')[0].childNodes[0].data
                thirdparty_label = thirdparty_label[0] # some label has extra symbols, like '5]', here remove the extra symbols
                thirdparty_label_list.append(thirdparty_label)
            
            data_dict[speaker]['text'][index] = text
            data_dict[speaker]['label_selfsentiment'][index] = label
            #user time is not gotten from eaf files, but from time synchro files in the process below
            data_dict[speaker]['speaker'][index] = speaker
            if(label == ' ' or label == ' E' or label == 'E'):
                data_dict[speaker]['has_label'][index] = 0
                print(speaker, index, ' No label', label)
            else:
                data_dict[speaker]['has_label'][index] = 1
            
            if(' ' in thirdparty_label_list or ' E' in thirdparty_label_list or 
               'E' in thirdparty_label_list or 'e' in thirdparty_label_list):
                data_dict[speaker]['has_all_thirdparty_label'][index] = 0
                print(speaker, index, ' Missing third party label', label)
            else:
                data_dict[speaker]['has_all_thirdparty_label'][index] = 1
            
            data_dict[speaker]['label_thirdpartysentiment'][index] = sum([int(t_l) for t_l in thirdparty_label_list]) / len(thirdparty_label_list) if data_dict[speaker]['has_all_thirdparty_label'][index] == 1 else -100
        
        #read synchro
        with open(os.path.join(synchro_root, group, speaker + '_annotation.csv'), 'r', errors='ignore') as csv_i:
            csv_reader = csv.reader(csv_i)
            #verify whehter the length of synchro is the same to that of transcript or annotation
            length_count = 0
            for s_i, row in enumerate(csv_reader):
                length_count += 1
            if(len(data_dict[speaker]['text']) != length_count - 1):
                print(group, speaker)
        
        with open(os.path.join(synchro_root, group, speaker + '_annotation.csv'), 'r', errors='ignore') as csv_i:
            csv_reader = csv.reader(csv_i)
            content2index = {}
            for s_i, row in enumerate(csv_reader):
                if(s_i == 0):
                    for c_i, c in enumerate(row):
                        content2index[c] = c_i
                else:
                    #confirm whether the start ts is the same as the ts in transcript
                    data_dict[speaker]['start_time_user'][s_i - 1] = row[content2index['start(user)[ms]']]
                    data_dict[speaker]['end_time_user'][s_i - 1] = row[content2index['end(user)[ms]']]
                    data_dict[speaker]['start_time_agent'][s_i - 1] = row[content2index['start(agent)[ms]']]
                
        #then read au data from open face results
        #here use glob only for confirming whether seg clips is the same number as transcripts/annotations
        seg_dirs = glob.glob(os.path.join(au_root, speaker, '*'))
        if(len(seg_dirs) != len(transcript_contents)):
            print(speaker)
        
        for seg_i in range(len(seg_dirs)): #seg_dirs is formated as SPEAKER_NUM, so the NUM should be in order, so here use for with index but not for with list
            seg_dir = os.path.join(au_root, speaker, speaker + '_' + str(seg_i))
            seg_dir_name = seg_dir.split('\\')[-1]
            csv_temp = []
            au2index = {}
            if(os.path.exists(os.path.join(seg_dir, seg_dir_name + '.csv')) == False):
                print(speaker, seg_i, ' No open face csv')
                au_array = np.zeros((2, len(AU_list))) #basically no AU here, 2 is to avoid warning when computing mean or var
                data_dict[speaker]['has_AU'][seg_i] = 0
            else:
                with open(os.path.join(seg_dir, seg_dir_name + '.csv'), 'r') as csv_in:
                    csv_reader = csv.reader(csv_in)
                    for row in csv_reader:
                        csv_temp.append(row)
                for index, c in enumerate(csv_temp[0]):
                    if(c.strip(' ') in AU_list):
                        au2index[c.strip(' ')] = index
                au_array = np.zeros((len(csv_temp) - 1, len(AU_list)))
                for row_i, row in enumerate(csv_temp):
                    if(row_i == 0):
                        continue
                    else:
                        for au_i, au in enumerate(AU_list):
                            au_array[row_i - 1][au_i] = row[au2index[au]]
            data_dict[speaker]['AU'][seg_i] = au_array
            data_dict[speaker]['has_AU'][seg_i] = 1
        print(speaker + ' Over')

with open(os.path.join(save_root, 'Pre_HazumiNeedContents.pkl'), 'wb') as pkl_o:
    pickle.dump(data_dict, pkl_o)
with open(os.path.join(save_root, 'Pre_InfoHazumiNeedContents.pkl'), 'wb') as pkl_o:
    pickle.dump(info_dict, pkl_o)