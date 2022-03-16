import os
import torch
import pandas as pd
import soundfile as sf
import numpy as np
import subprocess
import librosa
import scipy.signal as spsig
from transformers import Wav2Vec2Processor
import sys
from tqdm import tqdm
import fairseq

sys.path.append('/data2/hzp/ABAW_VA_2022/code/preprocess')
from tools.base_worker import BaseWorker

class Wav2VecExtractor(object):
    ''' 抽取wav2vec特征, 输入音频路径, 输出npy数组, 每帧768d
    '''
    # def __init__(self, downsample=4, gpu=0, max_seg_len=60, use_asr_based_model=False):
    def __init__(self, gpu=0, max_seg_len=120, use_asr_based_model=False):
        '''
        - max_set_len: 一次送入语音模型的最大语音长度 (s)，在语音长度较长的情况下，将原始的语音帧序列切成多个片段
                       为确保后续流程的准确性，这里应该设为2的倍数
        '''
        # self.downsample = downsample
        self.device = torch.device('cuda:{}'.format(gpu))
        self.max_seg_len = max_seg_len
        ckpt_path = '/data2/hzp/ABAW_VA_2022/code/preprocess/tools/wav2vec_ckpt/checkpoint_20_120000.pt'
        if use_asr_based_model:
            print('[INFO] use asr based model')
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            # self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h").to(self.device)
            self.model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
            self.model = self.model[0]
        else:
            print('[INFO] use vanilla based model')
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
            # self.model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base").to(self.device)
            self.model, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
            self.model = self.model[0]
        self.model.eval()
        self.model.to(self.device)
        
    @staticmethod
    def read_audio(self, wav_path):
        '''
        return:
        - speech_list: [speech (max_len), speech (max_len), ...]
        - sr: sample rate
        '''
        speech, sr = sf.read(wav_path)
        speech_list = []
        if sr != 16000:
            speech = librosa.resample(speech, sr, 16000)
            sr = 16000
        s_idx = 0
        e_idx = int(sr * self.max_seg_len)
        while e_idx < len(speech):
            speech_list.append(speech[s_idx: e_idx])
            s_idx = e_idx
            e_idx += int(sr * self.max_seg_len)
        speech_list.append(speech[s_idx:])
        return speech_list, sr


    def __call__(self, wav):
        speech_list, sr = Wav2VecExtractor.read_audio(self, wav)
        ft_list = []
        for speech in tqdm(speech_list[:-1]):
            input_values = self.processor(speech, return_tensors="pt", sampling_rate=sr).input_values.to(self.device)
            with torch.no_grad():
                # ft = self.model(input_values).last_hidden_state
                ft = self.model(input_values, features_only=True)
                ft = ft['x']

            ft = torch.squeeze(ft).cpu().numpy()
            # 对于前面的片段，强制抽出的帧数与max_seg_len对应，避免多段累积造成较大的误差
            num_frames = int(self.max_seg_len / 0.02) # wav2vec为0.02s一帧
            if ft.shape[0] < num_frames:
                pad_ft = []
                pad_num = num_frames - ft.shape[0]
                for i in range(pad_num):
                    pad_ft.append(ft[-1])
                pad_ft = np.stack(pad_ft)
                ft = np.concatenate((ft, pad_ft), axis=0)
            else:
                ft = ft[:num_frames]
            ft_list.append(ft)

        # 最后一个片段，在这里不强制对齐时间
        speech = speech_list[-1]
        input_values = self.processor(speech, return_tensors="pt", sampling_rate=sr).input_values.to(self.device)
        with torch.no_grad():
            # ft = self.model(input_values).last_hidden_state
            ft = self.model(input_values, features_only=True)
            ft = ft['x']

        ft = torch.squeeze(ft).cpu().numpy()
        ft_list.append(ft)
        
        # 拼接
        audio_ft = np.concatenate(ft_list, axis=0)
        return audio_ft
    

if __name__ == '__main__':
    # wav2vec_extract = Wav2VecExtractor(downsample=-1, gpu=5)
    wav2vec_extract = Wav2VecExtractor(gpu=0)
    # audio_path = "/data9/hzp/ABAW_VA_2022/processed_data/audios/10-60-1280x720.wav" # 该数据对应的图像有2502帧?
    audio_path = "/data2/hzp/ABAW_VA_2022/processed_data/audios/9-15-1920x1080.wav" # 数据集里最长的语音1581.0235s，26分钟，只能切开处理
    ft = wav2vec_extract(audio_path)
    print(ft)
    # print(ft.shape) # (4168, 768) # wav2vec 0.02s一帧
    print(ft.shape) # (79050, 768) # wav2vec 0.02s一帧
