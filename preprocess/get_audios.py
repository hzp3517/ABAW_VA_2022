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

class AudioSplitorTool(BaseWorker):
    def __init__(self, logger=None):
        super().__init__()
        self.logger = logger

    def __call__(self, video_path, save_path):
        if not os.path.exists(save_path):
            _cmd = "ffmpeg -i {} -vn -f wav -acodec pcm_s16le -ac 1 -ar 16000 {} -y > /dev/null 2>&1".format(video_path, save_path)
            os.system(_cmd)
        return save_path

# def get_audios(path_list):
#     get_audio = AudioSplitorTool()
#     audio_path = get_audio(path_list)



if __name__ == '__main__':
    video_dir = '/data9/datasets/Aff-Wild2/videos/'
    audio_save_dir = '/data9/hzp/ABAW_VA_2022/processed_data/audios'
    get_audio = AudioSplitorTool()
    mkdir(audio_save_dir)

    # video_path_list = [os.path.join(video_dir, i) for i in os.listdir(video_dir)]
    # video_name_list = [get_basename(i) for i in video_path_list]
    # save_path_list = [os.path.join(audio_save_dir, i + '.wav') for i in video_name_list]

    # for video, save in zip(video_path_list, save_path_list):
    #     get_audios([video, save])

    video_path_list = [os.path.join(video_dir, i) for i in os.listdir(video_dir)]
    # video_name_list = [get_basename(i) for i in video_path_list]

    for video_path in tqdm(video_path_list):
        video = get_basename(video_path)
        save_path = os.path.join(audio_save_dir, video + '.wav')
        audio_path = get_audio(video_path, save_path)


