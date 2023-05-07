# -*- coding: utf-8 -*-
"""
This script is to convert MP4 video files to wav files for subsequential process such as audio feature extractions
"""

import os
import glob

data_root = '../Data_ori/Hazumi/Video'
audio_root = '../Data_ori/Hazumi/Audio'
if(os.path.exists(audio_root) == False):
    os.mkdir(audio_root)

for group in ['1902', '1911']:
    group_dir = os.path.join(audio_root, group)
    if(os.path.exists(group_dir) == False):
        os.mkdir(group_dir)
    videos = glob.glob(os.path.join(data_root, group, '*'))
    for video in videos:
        video_name = video.split('\\')[-1].split('.')[0]
        output_name = os.path.join(group_dir, video_name + '.wav')
        os.system('ffmpeg -i ' + video + ' ' + output_name)