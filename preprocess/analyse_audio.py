'''
看看最长的一条语音有多长
'''
import os
import glob
from tqdm import tqdm
import torch
import pandas as pd
import soundfile as sf
import numpy as np
import subprocess
import librosa
import scipy.signal as spsig
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tools.base_worker import BaseWorker
import h5py
import ffmpeg


def get_basename(path):
    basename = os.path.basename(path)
    if os.path.isfile(path):
        basename = basename[:basename.rfind('.')]
    return basename

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

audio_info_list = []

audio_dir = '/data9/hzp/ABAW_VA_2022/processed_data/audios/'
audio_list = os.listdir(audio_dir)
audio_path_list = [os.path.join(audio_dir, i) for i in audio_list]

max_time = 0
max_name = ''
for audio in tqdm(audio_path_list):
    audio_info = {}
    audio_info['name'] = get_basename(audio)
    if 'duration' in ffmpeg.probe(audio)['format'].keys(): # 视频流存在
        time_length = float(ffmpeg.probe(audio)['format']['duration'])
        audio_info['duration'] = time_length
        
        if time_length >= max_time:
            max_time = time_length
            max_name = get_basename(audio)
    else: # 视频流异常导致没有'duration'
        audio_info['duration'] = 0.0

print(max_name)
print(max_time)



