'''
以单张人脸为单位加载归一化后的图像
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

class SingleFrameToyDataset(BaseDataset):
    # @staticmethod
    # def modify_commandline_options(parser, is_train=True):
    #     parser.add_argument('--norm_features', type=str, default='None', help='feature to normalize (on train set), split by comma, eg: "egemaps,vggface"')
    #     return parser
    
    def __init__(self, opt, set_name):
        ''' SingleFrame dataset
        Parameter:
        --------------------------------------
        set_name: [train, val, test]
        '''
        super().__init__(opt)
        self.root = '/data9/hzp/ABAW_VA_2022/processed_data/toy/'
        self.set_name = set_name
        self.mean = [0.485, 0.456, 0.406] # 后续修改
        self.std = [0.229, 0.224, 0.225] # 后续修改
        self.load_label()
        self.load_data_idx() # 加载存储图像的h5文件的索引
        assert len(self.target) == len(self.data)
        self.manual_collate_fn = False
        print(f"SingleFrame dataset {set_name} created with total length: {len(self)}")

    def load_label(self):
        label_path = os.path.join(self.root, 'targets/{}_frame_targets.h5'.format(self.set_name))
        label_h5f = h5py.File(label_path, 'r')
        self.frame_list = list(label_h5f.keys())
        # self.target = {}
        # for frame in self.frame_list:
        #     self.target[frame] = {}
        #     self.target[frame]['valence'] = label_h5f[frame]['valence'][()]
        #     self.target[frame]['arousal'] = label_h5f[frame]['arousal'][()]
        self.target_list = []
        for frame in self.frame_list:
            frame_dict = {}
            frame_dict['valence'] = label_h5f[frame]['valence'][()]
            frame_dict['arousal'] = label_h5f[frame]['arousal'][()]
            self.target_list.append(frame_dict)

    def load_data_idx(self):
        data_path = os.path.join(self.root, 'features/{}_frame_img_data.h5'.format(self.set_name))
        data_h5f = h5py.File(data_path, 'r')
        # self.data_idx = {}
        # for frame in self.frame_list:
        #     self.data_idx[frame] = data_h5f[frame]
        self.data_idx_list = []
        for frame in self.frame_list:
            self.data_idx_list.append(data_h5f[frame])

    def __getitem__(self, index):
        img = self.data_idx_list[index]['image']
        valence = self.target_list[index]['valence']
        arousal = self.target_list[index]['arousal']
        return {**{"data": img, "valence": valence, "arousal": arousal}}
    
    def __len__(self):
        return len(self.frame_list)


if __name__ == '__main__':
    import torch.utils.data.dataloader as DataLoader
    
    class Test:
        feature_set = 'None'
        
    opt = Test()
    
    a = SingleFrameToyDataset(opt, 'train')
    # iter_a = iter(a)
    # data1 = next(iter_a)
    # data2 = next(iter_a)
    # data3 = next(iter_a)
    
    dataloader = DataLoader.DataLoader(a, batch_size=3, shuffle = True)
    for i, data in enumerate(dataloader):
        print(data['data'])
        print(data['valence'])
        print(data['arousal'])
        if i >= 2:
            break