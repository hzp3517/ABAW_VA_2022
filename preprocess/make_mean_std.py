import h5py
import numpy as np
import torch
from tqdm import tqdm
import os

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


def cal_mean_std_on_trn(features_dir, feature):
    input_file = os.path.join(features_dir, 'train_{}.h5'.format(feature))
    mkdir(os.path.join(features_dir,  'mean_std_on_trn'))
    output_file = os.path.join(features_dir,  'mean_std_on_trn', '{}.h5'.format(feature)) #记录mean和std信息的h5文件路径
    h5f = h5py.File(output_file, 'w')
    in_data = h5py.File(input_file, 'r')
    feature_data = []
    for video in tqdm(list(in_data.keys())):
        feature_data.append(in_data[video]['fts'][()])
    feature_data = np.concatenate(feature_data, axis=0)
    mean_f = np.mean(feature_data, axis=0)
    std_f = np.std(feature_data, axis=0)
    std_f[std_f == 0.0] = 1.0
    group = h5f.create_group('train') #创建一个名为'train'的组
    group['mean'] = mean_f
    group['std'] = std_f
    print(mean_f.shape)
    return mean_f, std_f


if __name__ == '__main__':
    features_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features/'
    features = ['compare', 'egemaps']
    for ft in features:
        print('process feature:', ft)
        mean_f, std_f = cal_mean_std_on_trn(features_dir, ft) #返回值为np.array类型

