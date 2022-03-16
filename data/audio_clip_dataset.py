'''
每次读入几个视频的语音，并按设定的最大长度切成多个片段，并打乱顺序。
'''

import os
import glob
import copy
from tqdm import tqdm
import torch
import pandas as pd
import soundfile as sf
import numpy as np
import subprocess
import librosa
import scipy.signal as spsig
import h5py
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from transformers import Wav2Vec2Processor
from data.base_dataset import BaseIterableDataset

class AudioClipDataset(BaseIterableDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_read_num', type=int, default=50, help='max number of audios reading once.')
        parser.add_argument('--max_clip_length', type=int, default=60, help='max length of che cutted audio clip (second)')
        return parser

    def __init__(self, opt, set_name):
        ''' SingleFrame dataset
        Parameter:
        --------------------------------------
        set_name: [train, val, test, train_eval]
        '''
        super().__init__(opt)
        self.root = '/data2/hzp/ABAW_VA_2022/processed_data/'
        self.audio_path = os.path.join(self.root, 'audios')
        self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
        self.set_name = set_name
        self.max_read_num = opt.max_read_num
        self.max_clip_length = opt.max_clip_length
        self.label_fps = 30 # 标签频率每秒30帧
        self.set_name = 'train' if set_name == 'train_eval' else set_name
        self.load_label()
        self.set_name = set_name
        if set_name == 'train':
            self.shuffle = True
        else:
            self.shuffle = False
        
        self.manual_collate_fn = True
        print(f"Audio Clip dataset {set_name} created with total length: {len(self)}")

    
    def load_label(self):
        label_path = os.path.join(self.root, 'targets/{}_original_all_targets.h5'.format(self.set_name))
        label_h5f = h5py.File(label_path, 'r')
        self.vid_list = list(label_h5f.keys())
        self.target_dict = {} # key为vid，value为这个音频内所有clip的标签列表
        self.valid_dict = {}
        self.len = 0

        for video in self.vid_list:
            valence = label_h5f[video]['valence'][()]
            arousal = label_h5f[video]['arousal'][()]
            video_labels = np.stack([valence, arousal]).transpose((1, 0))
            video_valid_lst = label_h5f[video]['valid'][()]

            video_label_clip_lst, valid_clip_lst, num_clips = self.get_clip_label_lst(video_labels, video_valid_lst)
            self.target_dict[video] = video_label_clip_lst
            self.valid_dict[video] = valid_clip_lst
            self.len += num_clips


            # self.target_dict[video] = video_labels
            # self.valid_dict[video] = label_h5f[video]['valid'][()]
            # length = len(video_labels)
            # self.len += length

    def get_clip_label_lst(self, video_labels, video_valid_lst):
        '''
        input:
        - video_label: [[v1, a1], [v2, a2], [], ...]
        - video_valid_lst: [1, 1, 0, ...]
        return:
        - label_list: [[[v1, a1], [v2, a2], [], ...] (max_len), [[..]..] (max_len), ...]
        - video_valid_lst: [[1, 1, ...], [1, 0, ...], ...]
        - clip_num
        '''
        label_clip_lst = []
        valid_clip_lst = []
        s_idx = 0
        e_idx = int(self.label_fps * self.max_clip_length)
        while e_idx < len(video_labels):
            label_clip_lst.append(video_labels[s_idx: e_idx])
            valid_clip_lst.append(video_valid_lst[s_idx: e_idx])
            s_idx = e_idx
            e_idx += int(self.label_fps * self.max_clip_length)
        label_clip_lst.append(video_labels[s_idx:])
        valid_clip_lst.append(video_valid_lst[s_idx:])
        return label_clip_lst, valid_clip_lst, len(label_clip_lst)


    def read_audio(self, wav_path, num_clips):
        '''
        - wav_path
        - num_clips: the number of label clips of this video
        return:
        - speech_list: [speech (max_len), speech (max_len), ...]
        - sr: sample rate
        '''
        speech, sr = sf.read(wav_path)
        speech_list = []
        # len_speech = 0
        if sr != 16000:
            speech = librosa.resample(speech, sr, 16000)
            sr = 16000
        s_idx = 0
        e_idx = int(sr * self.max_clip_length)
        while e_idx < len(speech):
            speech_list.append(speech[s_idx: e_idx])
            # len_speech += int(sr * self.max_clip_length)
            s_idx = e_idx
            e_idx += int(sr * self.max_clip_length)
        speech_list.append(speech[s_idx:])
        # len_speech += len(speech[s_idx:])

        # 如果抽出的语音分成的片段数少于标签划分的片段数，或者最后一个片段中的帧数过少，则需要pad语音数据
        minimum_clip_len = 400
        speech_data = np.concatenate(speech_list)
        num_pad_frames = 16000 * (num_clips - 1) * self.max_clip_length + minimum_clip_len - len(speech_data)

        if (len(speech_list) < num_clips) or (len(speech_list) == num_clips and num_pad_frames > 0):
            pad_speech = []
            for i in range(num_pad_frames):
                pad_speech.append(speech_data[-1])
            pad_speech = np.stack(pad_speech)
            speech_data = np.concatenate((speech_data, pad_speech), axis=0)

            new_speech_list = []
            s_idx = 0
            e_idx = int(sr * self.max_clip_length)
            while e_idx < len(speech_data):
                new_speech_list.append(speech[s_idx: e_idx])
                s_idx = e_idx
                e_idx += int(sr * self.max_clip_length)
            new_speech_list.append(speech[s_idx:])
            speech_list = new_speech_list

        return speech_list, sr


    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # 单进程：一个进程处理全部样本
            vids = copy.deepcopy(self.vid_list)
        else:  # 多进程，在当前进程中
            # 划分工作空间
            per_worker = int(len(self.vid_list) / float(worker_info.num_workers))
            worker_id = worker_info.id
            vids = copy.deepcopy(self.vid_list[per_worker * worker_id: min(len(self.vid_list), per_worker * (worker_id+1))])

        if self.shuffle:
            np.random.shuffle(vids)

        video_group_list = [] # [[video_1, video_2, ...], [], ...]
        s_idx = 0
        e_idx = self.max_read_num
        total_video_num = len(self.vid_list)
        while e_idx < total_video_num:
            video_group_list.append(self.vid_list[s_idx: e_idx])
            s_idx = e_idx
            e_idx += self.max_read_num
        video_group_list.append(self.vid_list[s_idx:])

        for video_group in video_group_list:
            group_label_clip_lst = []
            group_speech_clip_lst = []
            group_valid_clip_lst = []

            group_num_clips = 0
            for video in video_group:
                video_label_clip_lst = self.target_dict[video]
                video_valid_clip_lst = self.valid_dict[video]
                num_clips = len(video_label_clip_lst)
                group_num_clips += num_clips
                group_label_clip_lst += video_label_clip_lst
                group_valid_clip_lst += video_valid_clip_lst
                wav_path = os.path.join(self.audio_path, video + '.wav')
                video_speech_clip_lst, sr = self.read_audio(wav_path, num_clips)
                group_speech_clip_lst += video_speech_clip_lst
            index = np.arange(group_num_clips)
            if self.shuffle:
                np.random.shuffle(index)

            all_speech, all_labels, all_valid = [], [], []
            for idx in index:
                all_speech.append(group_speech_clip_lst[idx])
                all_labels.append(group_label_clip_lst[idx])
                all_valid.append(group_valid_clip_lst[idx])

            for speech, label, valid in zip(all_speech, all_labels, all_valid):
                # speech = speech.astype(np.float32)
                input_values = self.processor(speech, return_tensors="pt", sampling_rate=sr).input_values
                input_values = input_values.float().squeeze()
                len_speech = len(speech.astype(np.float32))
                label = label.astype(np.float32)
                all_valid = valid.astype(np.float32)
                data = {'speech': input_values, 'len_speech': len_speech, 'label': torch.from_numpy(label), 'valid': torch.from_numpy(all_valid)} # img: ; label: [2,]
                yield data


    def __len__(self):
        return self.len

    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        len_speech = torch.tensor([sample['len_speech'] for sample in batch]) # 每段语音的有效长度的列表
        speech = pad_sequence([sample['speech'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        if self.set_name != 'test':
            label = pad_sequence([sample['label'] for sample in batch], padding_value=torch.tensor(-5.0), batch_first=True)
            valid = pad_sequence([sample['valid'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        return {
            'len_speech': len_speech, # (bs,)
            'speech': speech.float(), # (bs, len_speech)
            'label': label.float(), # (bs, max_clip_length*30, 2)
            'valid': valid.float() # (bs, max_clip_length*30)
        } if self.set_name != 'test' else {
            'len_speech': len_speech,
            'speech': speech.float()
        }




if __name__ == '__main__':
    from data import create_dataset_with_args
    # import torch.utils.data.dataloader as DataLoader

    class Test:
        max_read_num = 10
        max_clip_length = 60
        dataset_mode = 'audio_clip'
        batch_size = 3
        num_threads = 0
        max_dataset_size = float("inf")
        serial_batches = True

    opt = Test()

    # dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset_with_args(opt, set_name=['train'])[0]  # create a dataset given opt.dataset_mode and other options

    for i, data in enumerate(dataset):
        print(data['len_speech'].shape) # (bs,)
        print(data['speech'].shape) # (bs, len_speech)
        print(data['label'].shape) # (bs, max_clip_length*30)
        print(data['valid'].shape) # (bs, max_clip_length*30)

        print(data['len_speech'])
        print(data['speech'])
        print(data['label'])
        print(data['valid'])
        # if i >= 1:
        #     break
        break
