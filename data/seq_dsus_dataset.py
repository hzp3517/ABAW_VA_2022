'''
加载特征的时候，对特征序列在时序上做降采样；标签长度不变。
'''
import os
import h5py
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import h5py
from tqdm import tqdm
import math
from data.base_dataset import BaseDataset

class SeqDsUsDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--norm_method', type=str, default='trn', choices=['batch', 'trn'], help='whether normalize method to use')
        parser.add_argument('--norm_features', type=str, default='None', help='feature to normalize, split by comma, eg: "egemaps,vggface"')
        parser.add_argument('--downsample_rate', type=int, default=1, help='pick 1 frame from `downsample_rate` frames')
        return parser


    def __init__(self, opt, set_name):
        ''' SingleFrame dataset
        Parameter:
        --------------------------------------
        set_name: [train, val, test, train_eval]
        '''
        super().__init__(opt)
        self.root = '/data9/hzp/ABAW_VA_2022/processed_data/'
        # self.root = '/data9/hzp/ABAW_VA_2022/processed_data/toy'

        self.feature_set = list(map(lambda x: x.strip(), opt.feature_set.split(',')))
        self.norm_method = opt.norm_method
        self.norm_features = list(map(lambda x: x.strip(), opt.norm_features.split(',')))
        self.set_name = set_name
        self.downsample_rate = opt.downsample_rate
        self.load_label()
        self.load_feature()
        self.manual_collate_fn = True
        print(f"Aff-Wild2 Sequential dataset {set_name} created with total length: {len(self)}")


    def normalize_on_trn(self, feature_name, features):
        '''
        features的shape：[seg_len, ft_dim]
        mean_f与std_f的shape：[ft_dim,]，已经经过了去0处理
        '''
        mean_std_file = h5py.File(os.path.join(self.root, 'features', 'mean_std_on_trn', feature_name + '.h5'), 'r')
        mean_trn = np.array(mean_std_file['train']['mean'])
        std_trn = np.array(mean_std_file['train']['std'])
        features = (features - mean_trn) / std_trn
        return features
    
    
    def normalize_on_batch(self, features):
        '''
        输入张量的shape：[bs, seq_len, ft_dim]
        mean_f与std_f的shape：[bs, 1, ft_dim]
        '''
        mean_f = torch.mean(features, dim=1).unsqueeze(1).float()
        std_f = torch.std(features, dim=1).unsqueeze(1).float()
        std_f[std_f == 0.0] = 1.0
        features = (features - mean_f) / std_f
        return features
    
    
    def load_label(self):
        set_name = 'train' if self.set_name == 'train_eval' else self.set_name
        label_path = os.path.join(self.root, 'targets/{}_valid_targets.h5'.format(set_name))
        label_h5f = h5py.File(label_path, 'r')
        self.video_list = list(label_h5f.keys())

        self.target_list = []
        for video in self.video_list:
            video_dict = {}
            if self.set_name != 'test':
                video_dict['valence'] = torch.from_numpy(label_h5f[video]['valence'][()]).float()
                video_dict['arousal'] = torch.from_numpy(label_h5f[video]['arousal'][()]).float()
                video_dict['length'] = label_h5f[video]['length'][()]
            else:
                video_dict['length'] = label_h5f[video]['length'][()]
            self.target_list.append(video_dict)


    def downsample(self, data_list, step):
        all_len = len(data_list)
        sub_data_list = np.array([data_list[i+int(step/2)] for i in range(0, all_len-int(step/2), step)])
        assert len(sub_data_list) > 0
        return sub_data_list


    def load_feature(self):
        self.feature_data = {}
        for feature_name in self.feature_set:
            self.feature_data[feature_name] = []
            set_name = 'train' if self.set_name == 'train_eval' else self.set_name
            feature_path = os.path.join(self.root, 'features/{}_{}.h5'.format(set_name, feature_name))
            feature_h5f = h5py.File(feature_path, 'r')
            feature_list = []
            for idx, video in enumerate(tqdm(self.video_list, desc='loading {} feature'.format(feature_name))):
                video_dict = {}
                video_dict['fts'] = feature_h5f[video]['fts'][()] #shape:(seg_len, ft_dim)
                # video_dict['pad'] = feature_h5f[video]['pad'][()]
                # video_dict['valid'] = feature_h5f[video]['valid'][()]
                assert len(video_dict['fts']) == int(self.target_list[idx]['length']), '\
                    Data Error: In feature {}, video_id: {}, frame does not match label frame'.format(feature_name, video)
                # normalize on trn:
                if (self.norm_method=='trn') and (feature_name in self.norm_features):
                    video_dict['fts'] = self.normalize_on_trn(feature_name, video_dict['fts'])
                self.feature_data[feature_name].append(video_dict)