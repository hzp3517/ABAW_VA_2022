'''
以video为单位加载数据，适用于时序模型
'''
import os
import h5py
import copy
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code')
from data.base_dataset import BaseDataset
from utils.bins import get_center_and_bounds


class SeqLateDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--a_features', type=str, default='None', help='feature to use, split by comma, eg: "egemaps,vggface"')
        parser.add_argument('--v_features', type=str, default='None', help='feature to use, split by comma, eg: "egemaps,vggface"')
        parser.add_argument('--norm_method', type=str, default='trn', choices=['batch', 'trn'], help='whether normalize method to use')
        parser.add_argument('--norm_features', type=str, default='None', help='feature to normalize, split by comma, eg: "egemaps,vggface"')
        return parser


    def __init__(self, opt, set_name):
        ''' Sequential Late Fusion Dataset
        Parameter:
        --------------------------------------
        set_name: [train, val, test, train_eval]
        '''
        super().__init__(opt)
        self.root = '/data2/hzp/ABAW_VA_2022/processed_data/'
        self.a_features = list(map(lambda x: x.strip(), opt.a_features.split(','))) if opt.a_features != 'None' and opt.a_features != 'none' else []
        self.v_features = list(map(lambda x: x.strip(), opt.v_features.split(','))) if opt.v_features != 'None' and opt.v_features != 'none' else []
        self.norm_method = opt.norm_method
        self.norm_features = list(map(lambda x: x.strip(), opt.norm_features.split(',')))
        self.set_name = set_name

        bin_centers, bin_bounds = get_center_and_bounds(opt.cls_weighted)
        self.bin_centers = dict([(key, np.array(value)) for key, value in bin_centers.items()])
        self.bin_bounds = dict([(key, np.array(value)) for key, value in bin_bounds.items()])

        self.load_label()
        self.load_feature()
        self.manual_collate_fn = True
        print(f"Aff-Wild2 Sequential Late Fusion dataset {set_name} created with total length: {len(self)}")
    
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
                for target in ['valence', 'arousal']:
                    bin_labels = torch.zeros((len(video_dict[target]), ), dtype=torch.long)
                    for b in range(22):
                        index = (video_dict[target] < self.bin_bounds[target][b+1]) & (video_dict[target] > self.bin_bounds[target][b])
                        bin_labels[index] = b
                    video_dict[target+'_cls'] = bin_labels #[L, ]
            else:
                video_dict['length'] = label_h5f[video]['length'][()]
            self.target_list.append(video_dict)

    def load_feature(self):
        '''
        create two dictionary:
        self.a_feature_data; self.v_feature_data
        if no corresponding features, the dictionary should be empty
        '''
        all_features = self.a_features + self.v_features
        self.a_feature_data = {}
        self.v_feature_data = {}

        for feature_name in all_features:
            if feature_name in self.a_features:
                self.a_feature_data[feature_name] = []
            else:
                self.v_feature_data[feature_name] = []
            set_name = 'train' if self.set_name == 'train_eval' else self.set_name
            feature_path = os.path.join(self.root, 'features/{}_{}.h5'.format(set_name, feature_name))
            feature_h5f = h5py.File(feature_path, 'r')
            for idx, video in enumerate(tqdm(self.video_list, desc='loading {} feature'.format(feature_name))):
                video_dict = {}
                video_dict['fts'] = feature_h5f[video]['fts'][()] #shape:(seg_len, ft_dim)
                assert len(video_dict['fts']) == int(self.target_list[idx]['length']), '\
                    Data Error: In feature {}, video_id: {}, frame does not match label frame'.format(feature_name, video)
                # normalize on trn:
                if (self.norm_method=='trn') and (feature_name in self.norm_features):
                    video_dict['fts'] = self.normalize_on_trn(feature_name, video_dict['fts'])
                if feature_name in self.a_features:
                    self.a_feature_data[feature_name].append(video_dict)
                else:
                    self.v_feature_data[feature_name].append(video_dict)

    def __getitem__(self, index):
        '''
        if modal `x` has no features, the `x_feature_list` should be None
        '''
        target_data = self.target_list[index]
        
        # if self.a_feature_data:
        a_feature_list = []
        a_feature_dims = []
        for feature_name in self.a_features:
            data = torch.from_numpy(self.a_feature_data[feature_name][index]['fts']).float()
            a_feature_list.append(data)
            a_feature_dims.append(self.a_feature_data[feature_name][index]['fts'].shape[1])
        a_feature_dims = torch.from_numpy(np.array(a_feature_dims)).long()
        # else:
            # a_feature_list = None
            # a_feature_dims = None

        # if self.v_feature_data:
        v_feature_list = []
        v_feature_dims = []
        for feature_name in self.v_features:
            data = torch.from_numpy(self.v_feature_data[feature_name][index]['fts']).float()
            v_feature_list.append(data)
            v_feature_dims.append(self.v_feature_data[feature_name][index]['fts'].shape[1])
        v_feature_dims = torch.from_numpy(np.array(v_feature_dims)).long()
        # else:
            # v_feature_list = None
            # v_feature_dims = None

        return {**{"a_feature_list": a_feature_list, "v_feature_list": v_feature_list, 
                    "a_feature_dims": a_feature_dims, "v_feature_dims": v_feature_dims, "video_id": self.video_list[index]},
                **target_data, **{"a_feature_names": self.a_features, "v_feature_names": self.v_features}}
    
    def __len__(self):
        return len(self.video_list)
    

    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        if modal `x` has no features, the `x_feature` should be None
        '''
        a_feature_num = len(batch[0]['a_feature_list'])
        if a_feature_num:
            a_feature = []
            for i in range(a_feature_num):
                feature_name = self.a_features[i]
                pad_ft = pad_sequence([sample['a_feature_list'][i] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
                pad_ft = pad_ft.float()
                # normalize on batch:
                if (self.norm_method=='batch') and (feature_name in self.norm_features):
                    pad_ft = self.normalize_on_batch(pad_ft)
                a_feature.append(pad_ft)
            a_feature = torch.cat(a_feature, dim=2) # pad_ft: (bs, seq_len, ft_dim), concat all the audio features
        else:
            a_feature = None

        v_feature_num = len(batch[0]['v_feature_list'])
        if v_feature_num:
            v_feature = []
            for i in range(v_feature_num):
                feature_name = self.v_features[i]
                pad_ft = pad_sequence([sample['v_feature_list'][i] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
                pad_ft = pad_ft.float()
                # normalize on batch:
                if (self.norm_method=='batch') and (feature_name in self.norm_features):
                    pad_ft = self.normalize_on_batch(pad_ft)
                v_feature.append(pad_ft)
            v_feature = torch.cat(v_feature, dim=2) # pad_ft: (bs, seq_len, ft_dim), concat all the audio features
        else:
            v_feature = None

        length = torch.tensor([sample['length'] for sample in batch])
        video_id = [sample['video_id'] for sample in batch]

        if self.set_name != 'test':
            arousal = pad_sequence([sample['arousal'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
            valence = pad_sequence([sample['valence'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
            arousal_cls = pad_sequence([sample['arousal_cls'] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
            valence_cls = pad_sequence([sample['valence_cls'] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
        
        a_feature_dims = batch[0]['a_feature_dims']
        a_feature_names = batch[0]['a_feature_names']
        v_feature_dims = batch[0]['v_feature_dims']
        v_feature_names = batch[0]['v_feature_names']

        # make mask
        batch_size = length.size(0)
        batch_max_length = torch.max(length)
        mask = torch.zeros([batch_size, batch_max_length]).float()
        for i in range(batch_size):
            mask[i][:length[i]] = 1.0
        
        return {
            'a_feature': None if a_feature==None else a_feature.float(),
            'v_feature': None if v_feature==None else v_feature.float(),
            'arousal': arousal.float(), 
            'valence': valence.float(),
            'arousal_cls': arousal_cls.long(),
            'valence_cls': valence_cls.long(),
            'mask': mask.float(),
            'length': length,
            'a_feature_dims': a_feature_dims,
            'v_feature_dims': v_feature_dims,
            'a_feature_names': a_feature_names,
            'v_feature_names': v_feature_names,
            'video_id': video_id
        } if self.set_name != 'test' else {
            'a_feature': None if a_feature==None else a_feature.float(),
            'v_feature': None if v_feature==None else v_feature.float(),
            'mask': mask.float(),
            'length': length,
            'a_feature_dims': a_feature_dims,
            'v_feature_dims': v_feature_dims,
            'a_feature_names': a_feature_names,
            'v_feature_names': v_feature_names,
            'video_id': video_id
        }

if __name__ == '__main__':
    from data import create_dataset_with_args

    class test:
        a_features = 'vggish'
        v_features = 'affectnet'
        dataroot = '/data9/hzp/ABAW_VA_2022/processed_data/'
        max_seq_len = 100
        norm_method = ''
        norm_features = ''
        cls_weighted = False
        dataset_mode = 'seq_late'
        batch_size = 3
        serial_batches = False
        num_threads = 0
        max_dataset_size = float('inf')
    
    opt = test()

    dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options
    for i, data in enumerate(dataset):
        print(data['a_feature'].shape)
        print(data['v_feature'].shape)
        print(data['a_feature_dims'])
        print(data['v_feature_dims'])
        print(data['a_feature_names'])
        print(data['v_feature_names'])
        print(data['mask'].shape)
        print(data['length'])
        print(torch.sum(data['mask'][0]), torch.sum(data['mask'][1]), torch.sum(data['mask'][2]))
        print(data['valence'].shape)
        print(data['arousal'].shape)
        print(data['video_id'])
        if i >= 0:
            break

