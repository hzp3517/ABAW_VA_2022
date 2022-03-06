'''
以video为单位加载数据，适用于滑窗的时序模型
'''
import os
import h5py
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
# from .base_dataset import BaseDataset

import sys
sys.path.append('/data8/hzp/ABAW_VA_2022/code')#
from data.base_dataset import BaseDataset#

class SeqSlideDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--win_len', type=int, default=300, help='window length of a segment') 
            # the win_len should smaller than the `max_position_embeddings` in the transformer model.
        parser.add_argument('--hop_len', default=50, type=int, help='step length of a segmentt')
        parser.add_argument('--norm_method', type=str, default='trn', choices=['batch', 'trn'], help='whether normalize method to use')
        parser.add_argument('--norm_features', type=str, default='None', help='feature to normalize, split by comma, eg: "egemaps,vggface"')
        return parser
    
    
    def __init__(self, opt, set_name, inference_batched=False):
        ''' Sequence Slide dataset
        Parameter:
        --------------------------------------
        set_name: [train, val, test, train_eval]
        inference_batched: whether send the batched data or single data when do inference
        因为slide segment的设计，val和test的时候，是将整段视频送入的，视频长度可能不一致，因此不能组batch
        '''
        super().__init__(opt)
        self.root = '/data9/hzp/ABAW_VA_2022/processed_data/'
        self.feature_set = list(map(lambda x: x.strip(), opt.feature_set.split(',')))
        self.norm_method = opt.norm_method
        self.norm_features = list(map(lambda x: x.strip(), opt.norm_features.split(',')))
        self.win_len = opt.win_len
        self.hop_len = opt.hop_len
        self.set_name = 'train' if set_name == 'train_eval' else set_name
        self.load_label()
        self.load_feature()
        self.set_name = set_name
        self.manual_collate_fn = False
        if set_name == 'train':
            self.feature_segments, self.label_segments = self.make_segments()
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
        label_path = os.path.join(self.root, 'targets/{}_valid_targets.h5'.format(self.set_name))
        label_h5f = h5py.File(label_path, 'r')
        self.video_list = list(label_h5f.keys())

        self.target_list = []
        for video in self.video_list:
            video_dict = {}
            if self.set_name != 'test':
                video_dict['valence'] = torch.from_numpy(label_h5f[video]['valence'][()]).float()
                video_dict['arousal'] = torch.from_numpy(label_h5f[video]['arousal'][()]).float()
                video_dict['length'] = label_h5f[video]['length'][()]
                video_dict['video_id'] = video
            else:
                video_dict['length'] = label_h5f[video]['length'][()]
                video_dict['video_id'] = video
            self.target_list.append(video_dict)


    def load_feature(self):
        self.feature_data = {}
        for feature_name in self.feature_set:
            self.feature_data[feature_name] = []
            feature_path = os.path.join(self.root, 'features/{}_{}.h5'.format(self.set_name, feature_name))
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
                video_dict['fts'] = torch.tensor(video_dict['fts'], dtype=torch.float32)
                self.feature_data[feature_name].append(video_dict)


    def pad_max_len(self, tensor, max_len):
        # tensor = torch.from_numpy(tensor)
        # tensor -> T*D
        if len(tensor) < max_len:
            if tensor.ndim == 1:
                tensor = torch.cat([tensor, torch.zeros(max_len-len(tensor))], dim=0)
            else:
                tensor = torch.cat([tensor, torch.zeros(max_len-len(tensor), tensor.size(1))], dim=0)
        return tensor


    def make_segments(self):
        all_ft_segments = []
        all_label_segments = []
        
        for idx in range(len(self.video_list)):
            label_dict = self.target_list[idx]
            feature = []
            feature_dims = []
            for feature_name in self.feature_data.keys(): # concat feature
                feature.append(self.feature_data[feature_name][idx]['fts'])
                feature_dims.append(self.feature_data[feature_name][idx]['fts'].shape[1])
            feature = torch.cat(feature, dim=-1)
            feature_dims = torch.from_numpy(np.array(feature_dims)).long()
            length = label_dict['length']
            stop_flag = False
            seg_id = 0
            for st in range(0, length, self.hop_len):
                ed = st + self.win_len
                if ed > length:
                    ed = length
                    stop_flag = True
        
                ft_seg = {
                    'feature': self.pad_max_len(feature[st: ed, :], self.win_len).float(),
                    'feature_dims': feature_dims
                }
        
                label_seg = {
                    'arousal': self.pad_max_len(label_dict['arousal'][st:ed], self.win_len),
                    'valence': self.pad_max_len(label_dict['valence'][st:ed], self.win_len),
                    'length': torch.as_tensor(ed - st).long(), # 该片段的长度
                    'video_id': label_dict['video_id'], 
                    'seg_id': seg_id, 
                    'win_len': self.win_len,
                    'hop_len': self.hop_len,
                    'video_length': length # 对应原视频的长度
                }
                seg_id += 1
                all_label_segments.append(label_seg)
                all_ft_segments.append(ft_seg)
                if stop_flag:
                    break
        return all_ft_segments, all_label_segments


    def __getitem__(self, index):
        '''
        注意：val的时候batch_size必须是1，否则就要手写collate_fn
        '''
        if self.set_name == 'train':
            ft_seg = self.feature_segments[index]
            label_seg = self.label_segments[index]
            length = self.label_segments[index]['length'].item()
            mask = torch.zeros(len(ft_seg['feature'])).float()
            mask[:length] = 1.0
            return {
                **ft_seg, **{'mask': mask}, **label_seg, **{"feature_names": self.feature_set}
            }
            
        elif self.set_name != 'test':
            feature = torch.cat([
                self.feature_data[feat_name][index]['fts'] for feat_name in self.feature_set
            ], dim=-1)
            feature_dims = []
            for feature_name in self.feature_data.keys(): # concat feature
                feature_dims.append(self.feature_data[feature_name][index]['fts'].shape[1])
            feature_dims = torch.from_numpy(np.array(feature_dims)).long()
            return {
                'feature': feature.float(),
                'feature_dims': feature_dims,
                'arousal': self.target_list[index]['arousal'].float(),
                'valence': self.target_list[index]['valence'].float(),
                'mask': torch.ones(feature.size(0)).float(),
                "feature_names": self.feature_set, 
                'length': torch.tensor(self.target_list[index]['length']).long(),
                'video_id': self.target_list[index]['video_id']
            }
        else:
            feature = torch.cat([
                self.feature_data[feat_name][index]['fts'] for feat_name in self.feature_set
            ], dim=-1)
            feature_dims = []
            for feature_name in self.feature_data.keys(): # concat feature
                feature_dims.append(self.feature_data[feature_name][index]['fts'].shape[1])
            feature_dims = torch.from_numpy(np.array(feature_dims)).long()
            return {
                'feature': feature.float(),
                'feature_dims': feature_dims,
                'mask': torch.ones(feature.size(0)).long(),
                "feature_names": self.feature_set, 
                'length': torch.tensor(self.target_list[index]['length']).long(),
                'video_id': self.target_list[index]['video_id']
            }
            

    def __len__(self):
        return len(self.feature_segments) if self.set_name == 'train' else len(self.video_list)
    
    


if __name__ == '__main__':
    from data import create_dataset_with_args
    
    class test:
        feature_set = 'vggish'
        dataroot = '/data9/hzp/ABAW_VA_2022/processed_data/'
        win_len = 300
        hop_len = 50
        norm_method = ''
        norm_features = ''
        batch_size = 3
        serial_batches = True
        num_threads = 0
        dataset_mode = 'seq_slide'
        max_dataset_size = float("inf")
    
    opt = test()
    dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options

    for i, data in enumerate(dataset):
        print(data.keys())
        print(data['feature'])
        print(data['arousal'])
        print(data['valence'])
        print(data['mask'])
        print(data['length'])
        print(data['feature_names'])
        print(data['feature_dims'])
        print(data['video_id'])
        
        if i >= 0:
            break
