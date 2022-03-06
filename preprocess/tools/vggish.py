import os
import torch
import pandas as pd
import soundfile as sf
import numpy as np
import subprocess
import librosa
import scipy.signal as spsig
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import sys


sys.path.append('/data8/hzp/ABAW_VA_2022/code/preprocess')
from tools.base_worker import BaseWorker

class VggishExtractor(BaseWorker):
    ''' 抽取vggish特征, 输入音频路径, 输出npy数组, 每帧128d
    '''
    def __init__(self, seg_len=1/6, step_size=1/6, device=0):
        ''' Vggish feature extractor
            seg_len: window size(with expansion of 1s, padding 0.5s at both sides)
            step_size: step size of each window
            device: GPU number
        '''
        super().__init__()
        self.seg_len = seg_len
        self.step_size = step_size
        self.device = torch.device(f'cuda:{device}')
        self.model = self.get_pretrained_vggish()
    
    def read_wav(self, wav_path):
        wav_data, sr = sf.read(wav_path, dtype='int16')
        return wav_data, sr
    
    def get_pretrained_vggish(self):
        model = torch.hub.load('harritaylor/torchvggish', 'vggish')
        model.eval()
        model.postprocess = False
        model.device = self.device
        model.to(self.device)
        return model
    
    def get_vggish_segment(self, wav_data, sr, timestamps):
        block_len = int(0.98 * sr) ## vggish block is 0.96s, add some padding
        self.seg_len = int(self.seg_len * sr) 
        pad_context = (block_len - self.seg_len) // 2
        ans = []
        for timestamp in timestamps:
            timestamp = int(timestamp * sr)

            if timestamp >= len(wav_data) + pad_context: # 提供的部分音频长度比label的timestamp短
                cur_time_wav_data = np.array([wav_data[-1]] * block_len)
            else:                                        # 正常情况, timestamp的时间没超过audio_length
                left_padding = np.array(max(0, (pad_context - timestamp)) * [wav_data[0]]) #以第0时刻的值填补左侧
                right_padding = np.array(max(0, (timestamp + self.seg_len + pad_context) - len(wav_data)) * [wav_data[-1]]) #以最后一个时刻的值填补右侧
                target_data = wav_data[max(0, timestamp-pad_context): timestamp + self.seg_len + pad_context]
                cur_time_wav_data = np.concatenate([left_padding, target_data, right_padding])
                cur_time_wav_data = np.array(cur_time_wav_data)
            ans.append(cur_time_wav_data)
        return np.array(ans)

    def __call__(self, wav_path):
        wav_data, sr = self.read_wav(wav_path) #读入语音文件
        time_length = len(wav_data) / sr #len(wav_data):2377352, sr:16000. time_length=148.5845
        timestamps = [self.step_size * n for n in range(int(time_length / self.step_size))] #列表中的元素即为所有timestamp时刻（每个1/6s时间段的起始时刻）：[0, 1/6, 2/6, ..., 148+(2/6)]
        segments = self.get_vggish_segment(wav_data, sr, timestamps)
        vggish_feature = list(map(lambda x: self.model.forward(x, sr).cpu().detach().numpy(), segments)) #self.model.forward就是用github上vggish实现中的VGGish类的forward方法，先抽梅尔谱，然后过VGG
        vggish_feature = np.array(vggish_feature).squeeze()
        # self.print(f'Extract vggish from {wav_path}: {vggish_feature.shape}')  
        if len(vggish_feature) < 2 or vggish_feature.shape[0] == 0:
            return None

        return vggish_feature


if __name__ == '__main__':
    vggish_extract = VggishExtractor(seg_len=1/30, step_size=1/30, device=0)
    audio_path = "/data9/hzp/ABAW_VA_2022/processed_data/audios/10-60-1280x720.wav" # (2501, 128)，该数据对应的图像有2502帧
    ft = vggish_extract(audio_path)
    print(ft)
    print(ft.shape)