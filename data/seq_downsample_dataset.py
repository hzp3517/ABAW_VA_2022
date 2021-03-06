'''
读入数据的时候在时序上降采样，每n帧做个平均算作一个时刻。inference时不变。
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

class SeqDownsampleDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--norm_method', type=str, default='trn', choices=['batch', 'trn'], help='whether normalize method to use')
        parser.add_argument('--norm_features', type=str, default='None', help='feature to normalize, split by comma, eg: "egemaps,vggface"')
        parser.add_argument('--downsample_rate', type=int, default=1, help='combine `downsample_rate` frames in one timestamp when training')
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
        for video in tqdm(self.video_list, desc='loading target'):
            video_dict = {}
            if self.set_name != 'test':
                valence = label_h5f[video]['valence'][()]
                arousal = label_h5f[video]['arousal'][()]
                length = label_h5f[video]['length'][()]
                if self.set_name == 'train': # downsample
                    new_valence, new_arousal = [], []
                    length = math.ceil(length * 1.0 / self.downsample_rate)
                    if length > 1:
                        start_list = np.arange(length - 1) * self.downsample_rate
                        end_list = np.arange(length - 1) * self.downsample_rate + self.downsample_rate
                        for s_idx, e_idx in zip(start_list, end_list):
                            valence_mean = np.mean(valence[s_idx: e_idx])
                            arousal_mean = np.mean(arousal[s_idx: e_idx])
                            new_valence.append(valence_mean)
                            new_arousal.append(arousal_mean)
                    s_idx = (length - 1) * self.downsample_rate
                    new_valence.append(np.mean(valence[s_idx:]))
                    new_arousal.append(np.mean(arousal[s_idx:]))
                    valence = np.array(new_valence)
                    arousal = np.array(new_arousal)
                video_dict['valence'] = torch.from_numpy(valence).float()
                video_dict['arousal'] = torch.from_numpy(arousal).float()
                video_dict['length'] = length
                assert len(video_dict['valence']) == video_dict['length']
                assert len(video_dict['arousal']) == video_dict['length']
            else:
                video_dict['length'] = label_h5f[video]['length'][()]
                
            self.target_list.append(video_dict)
            
            
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
                fts = feature_h5f[video]['fts'][()] # shape:(seg_len, ft_dim)
                # normalize on trn:
                if (self.norm_method=='trn') and (feature_name in self.norm_features):
                    fts = self.normalize_on_trn(feature_name, fts)
                if self.set_name == 'train': # downsample
                    new_fts = []
                    length = len(fts)
                    length = math.ceil(length * 1.0 / self.downsample_rate)
                    
                    if length > 1:
                        start_list = np.arange(length - 1) * self.downsample_rate
                        end_list = np.arange(length - 1) * self.downsample_rate + self.downsample_rate
                        for s_idx, e_idx in zip(start_list, end_list):
                            fts_mean = np.mean(fts[s_idx: e_idx], axis=0)
                            new_fts.append(fts_mean)
                    s_idx = (length - 1) * self.downsample_rate
                    new_fts.append(np.mean(fts[s_idx:], axis=0))
                    fts = np.stack(new_fts, axis=0)
                    
                video_dict['fts'] = fts
                assert len(video_dict['fts']) == int(self.target_list[idx]['length']), '\
                    Data Error: In feature {}, video_id: {}, frame does not match label frame'.format(feature_name, video)

                self.feature_data[feature_name].append(video_dict)
                
                
    def __getitem__(self, index):
        target_data = self.target_list[index]
        feature_list = []
        feature_dims = []
        for feature_name in self.feature_set:
            data = torch.from_numpy(self.feature_data[feature_name][index]['fts']).float()
            feature_list.append(data)
            feature_dims.append(self.feature_data[feature_name][index]['fts'].shape[1])
        feature_dims = torch.from_numpy(np.array(feature_dims)).long()
        return {**{"feature_list": feature_list, "feature_dims": feature_dims, "video_id": self.video_list[index]},
                **target_data, **{"feature_names": self.feature_set}}
        
        
    def __len__(self):
        return len(self.video_list)
    
    
    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        '''
        feature_num = len(batch[0]['feature_list'])
        feature = []
        for i in range(feature_num):
            feature_name = self.feature_set[i]
            pad_ft = pad_sequence([sample['feature_list'][i] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
            pad_ft = pad_ft.float()
            # normalize on batch:
            if (self.norm_method=='batch') and (feature_name in self.norm_features):
                pad_ft = self.normalize_on_batch(pad_ft)
            feature.append(pad_ft)
        feature = torch.cat(feature, dim=2) # pad_ft: (bs, seq_len, ft_dim)，将各特征拼接起来

        length = torch.tensor([sample['length'] for sample in batch])
        video_id = [sample['video_id'] for sample in batch]

        if self.set_name != 'test':
            arousal = pad_sequence([sample['arousal'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
            valence = pad_sequence([sample['valence'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        
        feature_dims = batch[0]['feature_dims']
        feature_names = batch[0]['feature_names']
        # make mask
        batch_size = length.size(0)
        batch_max_length = torch.max(length)
        mask = torch.zeros([batch_size, batch_max_length]).float()
        for i in range(batch_size):
            mask[i][:length[i]] = 1.0
        
        return {
            'feature': feature.float(), 
            'arousal': arousal.float(), 
            'valence': valence.float(),
            'mask': mask.float(),
            'length': length,
            'feature_dims': feature_dims,
            'feature_names': feature_names,
            'video_id': video_id
        } if self.set_name != 'test' else {
            'feature': feature.float(), 
            'mask': mask.float(),
            'length': length,
            'feature_dims': feature_dims,
            'feature_names': feature_names,
            'video_id': video_id
        }
        
        
if __name__ == '__main__':
    class test:
        feature_set = 'denseface'
        dataroot = '/data9/hzp/ABAW_VA_2022/processed_data/'
        max_seq_len = 100
        norm_method = ''
        norm_features = ''
        downsample_rate = 3
        serial_batches = True
    
    opt = test()
    a = SeqDownsampleDataset(opt, 'train')

    iter_a = iter(a)
    data1 = next(iter_a)
    data2 = next(iter_a)
    data3 = next(iter_a)
    batch_data = a.collate_fn([data1, data2, data3])
    print(batch_data.keys())
    print(batch_data['feature'].shape)
    print(batch_data['arousal'].shape)
    print(batch_data['valence'].shape)
    print(batch_data['mask'].shape)
    print(batch_data['length'])
    print(torch.sum(batch_data['mask'][0]), torch.sum(batch_data['mask'][1]), torch.sum(batch_data['mask'][2]))
    print(batch_data['feature_names'])
    print(batch_data['feature_dims'])
    print(batch_data['video_id'])



    # print(data['feature'].shape)
    # print(data['feature_lens'])
    # print(data['feature_names'])
    # print(data['length'])