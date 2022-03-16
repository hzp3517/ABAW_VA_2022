'''
在训练的时候扩增降采样的数据
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

class SeqAugDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--norm_method', type=str, default='trn', choices=['batch', 'trn'], help='whether normalize method to use')
        parser.add_argument('--norm_features', type=str, default='None', help='feature to normalize, split by comma, eg: "egemaps,vggface"')
        parser.add_argument('--ds_list', type=str, default='3,5', help='downsample rate list, split by comma, eg: "3,5,7". if do not augment data, input "None"')
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
        if opt.ds_list == 'None' or opt.ds_list == '':
            self.ds_list = []
        else:
            self.ds_list = list(map(lambda x: int(x.strip()), opt.ds_list.split(',')))
        self.load_label()
        self.load_feature()
        self.manual_collate_fn = True
        print(f"Aff-Wild2 Sequential Augmented dataset {set_name} created with total length: {len(self)}")
        
        
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
    
    
    def make_aug_data(self, data_list, step):
        all_len = len(data_list)
        sub_data_list = np.array([data_list[i+int(step/2)] for i in range(0, all_len-int(step/2), step)])
        assert len(sub_data_list) > 0
        return sub_data_list
    
    
    def load_label(self):
        set_name = 'train' if self.set_name == 'train_eval' else self.set_name
        label_path = os.path.join(self.root, 'targets/{}_valid_targets.h5'.format(set_name))
        label_h5f = h5py.File(label_path, 'r')
        self.original_video_list = list(label_h5f.keys())
        self.aug_video_list = []

        self.target_list = []
        for video in self.original_video_list:
            video_dict = {}
            if self.set_name != 'test':
                video_dict['valence'] = torch.from_numpy(label_h5f[video]['valence'][()]).float()
                video_dict['arousal'] = torch.from_numpy(label_h5f[video]['arousal'][()]).float()
                video_dict['length'] = label_h5f[video]['length'][()]
                if self.set_name == 'train': # augment
                    for ds_rate in self.ds_list:
                        sub_video_dict = {}
                        sub_video_dict['valence'] = torch.from_numpy(self.make_aug_data(video_dict['valence'], ds_rate)).float()
                        sub_video_dict['arousal'] = torch.from_numpy(self.make_aug_data(video_dict['arousal'], ds_rate)).float()
                        sub_video_dict['length'] = len(sub_video_dict['valence'])
                        self.target_list.append(sub_video_dict)
                        self.aug_video_list.append('{}_ds{}'.format(video, ds_rate))
            else:
                video_dict['length'] = label_h5f[video]['length'][()]
            self.target_list.append(video_dict)
            self.aug_video_list.append(video)
            
            
    def load_feature(self):
        self.feature_data = {}
        for feature_name in self.feature_set:
            self.feature_data[feature_name] = []
            set_name = 'train' if self.set_name == 'train_eval' else self.set_name
            feature_path = os.path.join(self.root, 'features/{}_{}.h5'.format(set_name, feature_name))
            feature_h5f = h5py.File(feature_path, 'r')
            for idx, video in enumerate(tqdm(self.original_video_list, desc='loading {} feature'.format(feature_name))):
                video_dict = {}
                video_dict['fts'] = feature_h5f[video]['fts'][()] #shape:(seg_len, ft_dim)
                # normalize on trn:
                if (self.norm_method=='trn') and (feature_name in self.norm_features):
                    video_dict['fts'] = self.normalize_on_trn(feature_name, video_dict['fts'])
                # augment
                if self.set_name == 'train':
                    assert len(video_dict['fts']) == int(self.target_list[idx * (len(self.ds_list) + 1) + len(self.ds_list)]['length']), '\
                        Data Error: In feature {}, video_id: {}, frame does not match label frame'.format(feature_name, video)
                    for ds_rate in self.ds_list:
                        sub_video_dict = {}
                        sub_video_dict['fts'] = self.make_aug_data(video_dict['fts'], ds_rate)
                        self.feature_data[feature_name].append(sub_video_dict)
                else:
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
        return {**{"feature_list": feature_list, "feature_dims": feature_dims, "video_id": self.aug_video_list[index]},
                **target_data, **{"feature_names": self.feature_set}}
        
        
    def __len__(self):
        return len(self.aug_video_list)
    
    
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
        ds_list = '3,5,7'
        # ds_list = 'None'
    
    opt = test()
    a = SeqAugDataset(opt, 'train')

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