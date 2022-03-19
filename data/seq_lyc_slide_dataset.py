'''
lyc说的，训练的时候不扩增数据，只在验证的时候有重叠窗口
'''
import os
import h5py
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code')
from data.base_dataset import BaseDataset#
from utils.bins import get_center_and_bounds


class SeqLycSlideDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--win_len', type=int, default=250, help='window length of a segment') 
            # the win_len should smaller than the `max_position_embeddings` in the transformer model.
        parser.add_argument('--hop_len', default=50, type=int, help='step length of a segmentt')
        parser.add_argument('--norm_method', type=str, default='trn', choices=['batch', 'trn'], help='whether normalize method to use')
        parser.add_argument('--norm_features', type=str, default='None', help='feature to normalize, split by comma, eg: "egemaps,vggface"')
        return parser


    def __init__(self, opt, set_name):
        ''' MuseWild dataset
        Parameter:
        --------------------------------------
        set_name: [train, val, test]
        '''
        super().__init__(opt)
        self.root = '/data2/hzp/ABAW_VA_2022/processed_data/'
        # self.root = '/data2/hzp/ABAW_VA_2022/processed_data/toy'
        self.feature_set = list(map(lambda x: x.strip(), opt.feature_set.split(',')))
        self.norm_method = opt.norm_method
        self.norm_features = list(map(lambda x: x.strip(), opt.norm_features.split(',')))
        self.set_name = set_name

        bin_centers, bin_bounds = get_center_and_bounds(opt.cls_weighted)
        self.bin_centers = dict([(key, np.array(value)) for key, value in bin_centers.items()])
        self.bin_bounds = dict([(key, np.array(value)) for key, value in bin_bounds.items()])

        self.load_label()
        self.load_feature()
        self.win_len = opt.win_len
        self.hop_len = opt.hop_len
        self.feature_segments, self.label_segments = self.make_segments()
        
        if self.set_name == 'train':
            opt.serial_batches = False
        else:
            opt.serial_batches = True
        self.manual_collate_fn = False
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


    def pad_max_len(self, tensor, max_len):
        # tensor = torch.from_numpy(tensor)
        # tensor -> T*D
        if len(tensor) < max_len:
            if tensor.ndim == 1:
                tensor = torch.cat([tensor, torch.zeros(max_len-len(tensor))], dim=0)
            else:
                tensor = torch.cat([tensor, torch.zeros(max_len-len(tensor), tensor.size(1))], dim=0)
        return tensor


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
                video_dict['video_id'] = video
                for target in ['valence', 'arousal']:
                    bin_labels = torch.zeros((len(video_dict[target]), ), dtype=torch.long)
                    for b in range(22):
                        index = (video_dict[target] < self.bin_bounds[target][b+1]) & (video_dict[target] > self.bin_bounds[target][b])
                        bin_labels[index] = b
                    video_dict[target+'_cls'] = bin_labels #[L, ]
            else:
                video_dict['length'] = label_h5f[video]['length'][()]
            self.target_list.append(video_dict)

        self.video_len_list = [i['length'] for i in self.target_list]


    def load_feature(self):
        self.feature_data = {}
        feature_dims = []
        for feature_name in self.feature_set:
            self.feature_data[feature_name] = []
            set_name = 'train' if self.set_name == 'train_eval' else self.set_name
            feature_path = os.path.join(self.root, 'features/{}_{}.h5'.format(set_name, feature_name))
            feature_h5f = h5py.File(feature_path, 'r')
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

            feature_dims.append(self.feature_data[feature_name][0]['fts'].shape[1])
        self.feature_dims = torch.from_numpy(np.array(feature_dims)).long()

    
    def make_segments(self):
        all_ft_segments = []
        all_label_segments = []
        for index in range(len(self.video_list)):
            label_dict = self.target_list[index]
            feature = []
            for feature_name in self.feature_data.keys(): # concat feature
                fts = torch.from_numpy(self.feature_data[feature_name][index]['fts']).float()
                feature.append(fts)
            feature = torch.cat(feature, dim=-1)
            length = label_dict['length']
            stop_flag = False
            seg_id = 0

            real_hop_len = self.win_len if self.set_name == 'train' else self.hop_len

            # for st in range(0, length, self.hop_len):
            for st in range(0, length, real_hop_len):
                ed = st + self.win_len
                if ed > length:
                    ed = length
                    stop_flag = True

                label_seg = {
                    'valence': self.pad_max_len(label_dict['valence'][st:ed], self.win_len),
                    'arousal': self.pad_max_len(label_dict['arousal'][st:ed], self.win_len),
                    'seg_real_length': torch.as_tensor(ed - st).long(), # 片段实际长度（不算pad的部分）
                    'video_id': label_dict['video_id'], 
                    'seg_id': seg_id, 
                    'win_len': self.win_len,
                    # 'hop_len': self.hop_len,
                    'hop_len': real_hop_len,
                    'video_length': length
                }

                seg_id += 1
                all_label_segments.append(label_seg)
                ft_seg = self.pad_max_len(feature[st: ed, :], self.win_len)
                all_ft_segments.append(ft_seg)
                if stop_flag:
                    break
        return all_ft_segments, all_label_segments


    def __getitem__(self, index):
        ft_seg = self.feature_segments[index].float()
        label_seg = self.label_segments[index]
        length = self.label_segments[index]['seg_real_length'].item()
        mask = torch.zeros(len(ft_seg)).float()
        mask[:length] = 1.0

        return {
            **{"feature": ft_seg, "feature_dims": self.feature_dims, "mask": mask}, **label_seg, 
            **{"feature_names": self.feature_set}
        }


    def __len__(self):
        return len(self.feature_segments)


if __name__ == '__main__':
    # import torch.utils.data.dataloader as DataLoader
    from data import create_dataset_with_args

    class test:
        feature_set = 'denseface'
        win_len = 250
        hop_len = 50
        norm_method = ''
        norm_features = ''
        cls_weighted = False
        dataset_mode = 'seq_slide'
        batch_size = 3
        serial_batches = False
        num_threads = 0
        max_dataset_size = float('inf')
        dataset_mode = 'seq_lyc_slide'
    
    opt = test()

    dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options
    for i, data in enumerate(dataset):
        print(data['feature'].shape)
        print(data['feature_dims'])
        print(data['feature_names'])
        print(data['mask'].shape)
        print(data['valence'].shape)
        print(data['arousal'].shape)
        print(data['seg_real_length'])
        print(data['video_id'])
        print(data['seg_id'])
        print(data['win_len'])
        print(data['hop_len'])
        print(data['video_length'])
        if i >= 0:
            break

    print(val_dataset.dataset.video_len_list)