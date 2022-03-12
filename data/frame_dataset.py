'''
这里的video_id是原始的
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
import cv2

import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code')#
from data.base_dataset import BaseIterableDataset#

class FrameDataset(BaseIterableDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--img_type', type=str, default='color', choices=['color', 'gray'], help='color: 224*224*3, for vggface2; gray: 64*64, for denseface.')
        parser.add_argument('--read_length', type=int, default=200000, help='max number of reading data once.')
        parser.add_argument('--norm_type', type=str, default='norm', choices=['none', 'reduce_mean', 'norm'], help='reduce_mean: only reduce the mean, do not divide std, apply for the resnet_vggface2 model')
        parser.add_argument
        return parser

    def __init__(self, opt, set_name):
        ''' SingleFrame dataset
        Parameter:
        --------------------------------------
        set_name: [train, val, test, train_eval]
        '''
        super().__init__(opt)
        self.root = '/data2/hzp/ABAW_VA_2022/processed_data/'
        # self.dataset_root = '/data2/hzp/Aff-Wild2/'
        self.img_type = opt.img_type
        self.read_length = opt.read_length
        self.norm_type = opt.norm_type
        self.color_mean = [0.338, 0.362, 0.471]
        self.color_std = [0.230, 0.237, 0.273]
        self.gray_mean = 101.63449 # 在VA任务的训练集上计算得到
        self.gray_std = 59.74126
        self.set_name = 'train' if set_name == 'train_eval' else set_name
        self.load_label()
        self.load_img_key()
        self.set_name = set_name
        if set_name == 'train':
            self.shuffle = True
        else:
            self.shuffle = False
        self.manual_collate_fn = False
        print(f"Frame dataset {set_name} created with total length: {len(self)}")

    def load_label(self):
        label_path = os.path.join(self.root, 'targets/for_training_extractor/{}_targets.h5'.format(self.set_name))
        label_h5f = h5py.File(label_path, 'r')

        self.vid_list = list(label_h5f.keys())
        self.target_dict = {} # key为vid，value为这个视频内所有帧的label列表
        self.video_len = {}
        self.len = 0

        for video in self.vid_list:
            video_labels = label_h5f[video]['label'][()]
            self.target_dict[video] = video_labels
            length = len(video_labels)
            self.len += length
            self.video_len[video] = length

    def load_img_key(self):
        data_path = os.path.join(self.root, 'features/for_training_extractor/{}_imgs.h5'.format(self.set_name))
        data_h5f = h5py.File(data_path, 'r')
        self.data_key_dict = {}
        for video in self.vid_list:
            self.data_key_dict[video] = data_h5f[video]
            

    def process_img(self, img):
        '''
        - img: (112, 112, 3)
        return:
        - img: (3, 224, 224) or (64, 64)
        '''
        if self.img_type == 'color':
            img_size = 224
            img = cv2.resize(img, (img_size, img_size))
            img = img / 255.0
            if self.norm_type == 'reduce_mean':
                img = img - self.color_mean
            elif self.norm_type == 'norm':
                img = (img - self.color_mean) / self.color_std
            img = np.transpose(img, (2, 0, 1))
            assert img.shape == (3, img_size, img_size)
        else:
            img_size = 64
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (img_size, img_size))
            if self.norm_type == 'reduce_mean':
                img = img - self.gray_mean
            elif self.norm_type == 'norm':
                img = (img - self.gray_mean) / self.gray_std
            # img = img / 255.0
            img = np.expand_dims(img, axis=0)
            assert img.shape == (1, img_size, img_size)
        return img


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

        current_vids = []
        current_lens = []

        for vid in vids:
            #time1 = time.time()
            if sum(current_lens) < self.read_length:
                current_lens.append(self.video_len[vid])
                current_vids.append(vid)
            else:
                index = np.arange(sum(current_lens))
                if self.shuffle:
                    np.random.shuffle(index)
                all_images, all_labels = [], []
                for current_vid in current_vids:
                    images = self.data_key_dict[current_vid][()]
                    labels = self.target_dict[current_vid]
                    all_images.append(images)
                    all_labels.append(labels)
                all_images = np.concatenate(all_images, axis=0)[index] # np.array的用法，按index数组中每个元素对应的下标从原数组中取元素，得到一个新的数组
                all_labels = np.concatenate(all_labels, axis=0)[index]
                
                for img, label in zip(all_images, all_labels):
                    img = self.process_img(img).astype(np.float32)
                    label = label.astype(np.float32)
                    data = {'img': torch.from_numpy(img), 'label': torch.from_numpy(label)} # img: [3, 224, 224] or [1, 64, 64]; label: [2,]
                    yield data 

                current_vids, current_lens = [vid], [self.video_len[vid]]
                #time3 = time.time()

        # 最后一批数据也要送进去
        index = np.arange(sum(current_lens))
        if self.shuffle:
            np.random.shuffle(index)
        all_images, all_labels = [], []
        for current_vid in current_vids:
            images = self.data_key_dict[current_vid][()]
            labels = self.target_dict[current_vid]
            all_images.append(images)
            all_labels.append(labels)
        all_images = np.concatenate(all_images, axis=0)[index] # np.array的用法，按index数组中每个元素对应的下标从原数组中取元素，得到一个新的数组
        all_labels = np.concatenate(all_labels, axis=0)[index]
        
        for img, label in zip(all_images, all_labels):
            img = self.process_img(img).astype(np.float32)
            label = label.astype(np.float32)
            data = {'img': torch.from_numpy(img), 'label': torch.from_numpy(label)} # img: [3, 224, 224] or [1, 64, 64]; label: [2,]
            yield data 

        # current_vids, current_lens = [vid], [self.video_len[vid]]


    def __len__(self):
        return self.len


if __name__ == '__main__':
    import torch.utils.data.dataloader as DataLoader

    class Test:
        # img_type = 'color'
        img_type = 'gray'
        read_length = 400000
        norm_type = 'reduce_mean'

    opt = Test()

    a = FrameDataset(opt, 'train')

    dataloader = DataLoader.DataLoader(a, batch_size=3, shuffle = False)
    for i, data in enumerate(dataloader):
        print(data['img'].shape) # (3, 1, 64, 64)
        print(data['label'].shape) # (3, 2)
        if i > 1:
            break
