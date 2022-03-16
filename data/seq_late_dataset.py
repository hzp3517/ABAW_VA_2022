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
import h5py
from tqdm import tqdm
from data.base_dataset import BaseDataset


class SeqDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--a_features', type=str, default='None', help='feature to use, split by comma, eg: "egemaps,vggface"')
        parser.add_argument('--v_features', type=str, default='None', help='feature to use, split by comma, eg: "egemaps,vggface"')
        parser.add_argument('--l_features', type=str, default='None', help='feature to use, split by comma, eg: "egemaps,vggface"')
        return parser


    def __init__(self, opt, set_name):
        ''' MuseWild dataset
        Parameter:
        --------------------------------------
        set_name: [trn, val, tst]
        '''
        super().__init__(opt)
        self.root = '/data12/lrc/MUSE2021/h5_data/c3_muse_stress/'
        self.a_features = list(map(lambda x: x.strip(), opt.a_features.split(',')))
        self.v_features = list(map(lambda x: x.strip(), opt.v_features.split(',')))
        self.l_features = list(map(lambda x: x.strip(), opt.l_features.split(',')))

        self.set_name = set_name
        self.load_label()
        self.load_feature()
        self.manual_collate_fn = True
        print(f"MuseWild dataset {set_name} created with total length: {len(self)}")
    
    def load_label(self):
        partition_h5f = h5py.File(os.path.join(self.root, 'target', 'partition.h5'), 'r')
        self.seg_ids = sorted(partition_h5f[self.set_name])
        self.seg_ids = list(map(lambda x: str(x), self.seg_ids))
        label_h5f = h5py.File(os.path.join(self.root, 'target', '{}_target.h5'.format(self.set_name)), 'r')
        self.target = {}
        for _id in self.seg_ids:
            if self.set_name != 'tst':
                self.target[_id] = {
                    'arousal': torch.from_numpy(label_h5f[_id]['arousal'][()]).float(),
                    'valence': torch.from_numpy(label_h5f[_id]['valence'][()]).float(),
                    'length': torch.as_tensor(label_h5f[_id]['length'][()]).long(),
                    'timestamp': torch.from_numpy(label_h5f[_id]['timestamp'][()]).long(),
                }
            else:
                self.target[_id] = {
                    'length': torch.as_tensor(label_h5f[_id]['length'][()]).long(),
                    'timestamp': torch.from_numpy(label_h5f[_id]['timestamp'][()]).long(),
                }

    def load_feature(self):
        '''
        共创建三个字典：
        self.a_feature_data; self.v_feature_data; self.l_feature_data
        如果没有对应的特征，则该字典为空
        '''
        #a模态：
        self.a_feature_data = {}
        if self.a_features[0] != 'None':
            for feature_name in self.a_features:
                h5f = h5py.File(os.path.join(self.root, 'feature', '{}.h5'.format(feature_name)), 'r')
                feature_data = {}
                for _id in self.seg_ids:
                    feature_data[_id] = h5f[self.set_name][_id]['feature'][()]
                    # assert (h5f[self.set_name][_id]['timestamp'][()] == self.target[_id]['timestamp'].numpy()).all(), '\
                    assert len(h5f[self.set_name][_id]['timestamp'][()]) == len(self.target[_id]['timestamp']), '\
                        Data Error: In feature {}, seg_id: {}, timestamp does not match label timestamp'.format(feature_name, _id)
                self.a_feature_data[feature_name] = feature_data
        
        #v模态：
        self.v_feature_data = {}
        if self.v_features[0] != 'None':
            for feature_name in self.v_features:
                h5f = h5py.File(os.path.join(self.root, 'feature', '{}.h5'.format(feature_name)), 'r')
                feature_data = {}
                for _id in self.seg_ids:
                    feature_data[_id] = h5f[self.set_name][_id]['feature'][()]
                    # assert (h5f[self.set_name][_id]['timestamp'][()] == self.target[_id]['timestamp'].numpy()).all(), '\
                    assert len(h5f[self.set_name][_id]['timestamp'][()]) == len(self.target[_id]['timestamp']), '\
                        Data Error: In feature {}, seg_id: {}, timestamp does not match label timestamp'.format(feature_name, _id)
                self.v_feature_data[feature_name] = feature_data

        #l模态：
        self.l_feature_data = {}
        if self.l_features[0] != 'None':
            for feature_name in self.l_features:
                h5f = h5py.File(os.path.join(self.root, 'feature', '{}.h5'.format(feature_name)), 'r')
                feature_data = {}
                for _id in self.seg_ids:
                    feature_data[_id] = h5f[self.set_name][_id]['feature'][()]
                    # assert (h5f[self.set_name][_id]['timestamp'][()] == self.target[_id]['timestamp'].numpy()).all(), '\
                    assert len(h5f[self.set_name][_id]['timestamp'][()]) == len(self.target[_id]['timestamp']), '\
                        Data Error: In feature {}, seg_id: {}, timestamp does not match label timestamp'.format(feature_name, _id)
                self.l_feature_data[feature_name] = feature_data

    def __getitem__(self, index):
        '''
        如果某个模态无特征，返回值中对应的x_feature和x_feature_lens就设为None。
        '''
        seg_id = self.seg_ids[index]
        
        #a模态：
        if self.a_feature_data:
            a_feature_data = []
            a_feature_len = []
            for feature_name in self.a_features:
                a_feature_data.append(self.a_feature_data[feature_name][seg_id])
                a_feature_len.append(self.a_feature_data[feature_name][seg_id].shape[1])
            a_feature_data = torch.from_numpy(np.concatenate(a_feature_data, axis=1)).float()
            a_feature_data = a_feature_data.squeeze()
            a_feature_len = torch.from_numpy(np.array(a_feature_len)).long()
        else:
            a_feature_data = None
            a_feature_len = None

        #v模态：
        if self.v_feature_data:
            v_feature_data = []
            v_feature_len = []
            for feature_name in self.v_features:
                v_feature_data.append(self.v_feature_data[feature_name][seg_id])
                v_feature_len.append(self.v_feature_data[feature_name][seg_id].shape[1])
            v_feature_data = torch.from_numpy(np.concatenate(v_feature_data, axis=1)).float()
            v_feature_data = v_feature_data.squeeze()
            v_feature_len = torch.from_numpy(np.array(v_feature_len)).long()
        else:
            v_feature_data = None
            v_feature_len = None

        #l模态：
        if self.l_feature_data:
            l_feature_data = []
            l_feature_len = []
            for feature_name in self.l_features:
                l_feature_data.append(self.l_feature_data[feature_name][seg_id])
                l_feature_len.append(self.l_feature_data[feature_name][seg_id].shape[1])
            l_feature_data = torch.from_numpy(np.concatenate(l_feature_data, axis=1)).float()
            l_feature_data = l_feature_data.squeeze()
            l_feature_len = torch.from_numpy(np.array(l_feature_len)).long()
        else:
            l_feature_data = None
            l_feature_len = None

        target_data = self.target[seg_id]
        return {**{"a_feature": a_feature_data, "v_feature": v_feature_data, "l_feature": l_feature_data, 
                    "a_feature_lens": a_feature_len, "v_feature_lens": v_feature_len, "l_feature_lens": l_feature_len, "vid": seg_id},
                **target_data, **{"a_feature_names": self.a_features, "v_feature_names": self.v_features, "l_feature_names": self.l_features}} #**：对字典进行解引用
    
    def __len__(self):
        return len(self.seg_ids)
    
    def collate_fn(self, batch):
        '''
        Collate functions assume batch = [Dataset[i] for i in index_set]
        如果某个模态无特征，返回值中对应的x_feature和x_feature_lens就设为None。
        '''
        timestamp = pad_sequence([sample['timestamp'] for sample in batch], padding_value=torch.tensor(-1), batch_first=True)
        length = torch.tensor([sample['length'] for sample in batch])
        vid = [sample['vid'] for sample in batch]

        if self.set_name != 'tst':
            arousal = pad_sequence([sample['arousal'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
            valence = pad_sequence([sample['valence'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        
        # make mask
        batch_size = length.size(0)
        batch_max_length = torch.max(length)
        mask = torch.zeros([batch_size, batch_max_length]).float()
        for i in range(batch_size):
            mask[i][:length[i]] = 1.0

        a_feature = None if batch[0]['a_feature']==None else pad_sequence([sample['a_feature'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        v_feature = None if batch[0]['v_feature']==None else pad_sequence([sample['v_feature'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)
        l_feature = None if batch[0]['l_feature']==None else pad_sequence([sample['l_feature'] for sample in batch], padding_value=torch.tensor(0.0), batch_first=True)

        a_feature_lens = batch[0]['a_feature_lens']
        v_feature_lens = batch[0]['v_feature_lens']
        l_feature_lens = batch[0]['l_feature_lens']
        a_feature_names = batch[0]['a_feature_names']
        v_feature_names = batch[0]['v_feature_names']
        l_feature_names = batch[0]['l_feature_names']
        
        return {
            'a_feature': None if a_feature==None else a_feature.float(),
            'v_feature': None if v_feature==None else v_feature.float(),
            'l_feature': None if l_feature==None else l_feature.float(),
            'arousal': arousal.float(), 
            'valence': valence.float(),
            'timestamp': timestamp.long(),
            'mask': mask.float(),
            'length': length,
            'a_feature_lens': a_feature_lens,
            'v_feature_lens': v_feature_lens,
            'l_feature_lens': l_feature_lens,
            'a_feature_names': a_feature_names,
            'v_feature_names': v_feature_names,
            'l_feature_names': l_feature_names,
            'vid': vid
        } if self.set_name != 'tst' else {
            'a_feature': None if a_feature==None else a_feature.float(),
            'v_feature': None if v_feature==None else v_feature.float(),
            'l_feature': None if l_feature==None else l_feature.float(),
            'timestamp': timestamp.long(),
            'mask': mask.float(),
            'length': length,
            'a_feature_lens': a_feature_lens,
            'v_feature_lens': v_feature_lens,
            'l_feature_lens': l_feature_lens,
            'a_feature_names': a_feature_names,
            'v_feature_names': v_feature_names,
            'l_feature_names': l_feature_names,
            'vid': vid
        }

if __name__ == '__main__':
    class test:
        #feature_set = 'bert,vggface,vggish'
        feature_set = 'None'
        #a_features = 'vggish'
        a_features = 'None'
        v_features = 'vggface'
        l_features = 'bert'
        dataroot = '/data12/lrc/MUSE2021/h5_data/c3_muse_stress/'
        max_seq_len = 100
    
    opt = test()
    a = MuseLateStressDataset(opt, 'trn')
    iter_a = iter(a)
    data1 = next(iter_a)
    data2 = next(iter_a)
    data3 = next(iter_a)
    batch_data = a.collate_fn([data1, data2, data3])
    print(batch_data.keys())
    #print(batch_data['a_feature'].shape)
    print(batch_data['v_feature'].shape)
    print(batch_data['l_feature'].shape)
    print(batch_data['arousal'].shape)
    print(batch_data['valence'].shape)
    print(batch_data['mask'].shape)
    print(batch_data['length'])
    print(torch.sum(batch_data['mask'][0]), torch.sum(batch_data['mask'][1]), torch.sum(batch_data['mask'][2]))
    print(batch_data['a_feature_names'])
    print(batch_data['v_feature_names'])
    print(batch_data['l_feature_names'])
    print(batch_data['a_feature_lens'])
    print(batch_data['v_feature_lens'])
    print(batch_data['l_feature_lens'])
    print(batch_data['vid'])
    # print(data['feature'].shape)
    # print(data['feature_lens'])
    # print(data['feature_names'])
    # print(data['length'])

