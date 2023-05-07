# -*- coding: utf-8 -*-
"""
This script is for segmenting audio files by utterance index based on start and end time
"""

import os
import glob
import pickle
from pydub import AudioSegment

audio_root = '../Data_ori/Hazumi/Audio'
audio_seg_root = '../Data_ori/Hazumi/AudioSeg'
data_root = '../Data/Hazumi'

with open(os.path.join(data_root, 'Pre_HazumiNeedContents.pkl'), 'rb') as pkl_in:
    data_dict = pickle.load(pkl_in)

for group in ['1902', '1911']:
    group_dir = os.path.join(audio_seg_root, group)
    if(os.path.exists(group_dir) == False):
        os.mkdir(group_dir)
    wavs = glob.glob(os.path.join(audio_root, group, '*.wav'))
    for wav in wavs:
        speaker = wav.split('\\')[-1].split('.wav')[0]
        speaker_dir = os.path.join(group_dir, speaker)
        if(os.path.exists(speaker_dir) == False):
            os.mkdir(speaker_dir)
        #read wav file and get params
        wav_read = AudioSegment.from_wav(wav)
        
        speaker_start_times = data_dict[speaker]['start_time_user']
        speaker_end_times = data_dict[speaker]['end_time_user']
        has_label = data_dict[speaker]['has_label']
        for index in speaker_start_times:
            if(has_label[index] == 0):
                continue
            else:
                st = int(speaker_start_times[index])
                ed = int(speaker_end_times[index])
                seg_wav = wav_read[st: ed]
                if(data_dict[speaker] == '1911M3002'):
                    print(st, ed)
                seg_wav.export(os.path.join(speaker_dir, speaker + '_' + str(index) + '.wav'), format='wav')
        print(speaker + ' Over')