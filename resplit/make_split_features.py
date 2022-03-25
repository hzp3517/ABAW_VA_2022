'''
{
    [new_video_id]: 
    {
        'fts': [ft, ft, ...], # (video_len, dim)
    }
}
'''

import os
import h5py
import numpy as np
import glob
from tqdm import tqdm
import json

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass
    
def get_basename(path):
    basename = os.path.basename(path)
    if os.path.isfile(path):
        basename = basename[:basename.rfind('.')]
    return basename

target_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets'
features_dir  ='/data2/hzp/ABAW_VA_2022/processed_data/features/'
resplit_target_dir = os.path.join(target_dir, 'resplit')
resplit_features_dir = os.path.join(features_dir, 'resplit')
mkdir(resplit_features_dir)

feature_names = ['affectnet', 'vggish', 'wav2vec', 'compare', 'egemaps', 'FAU', 'FAU_situ', 'head_pose', 'eye_gaze']

for feature_name in feature_names:
    print('--------------process {}-------------'.format(feature_name))
    ft_dict = {}
    ft_train_h5_path = os.path.join(features_dir, 'train_{}.h5'.format(feature_name))
    ft_val_h5_path = os.path.join(features_dir, 'val_{}.h5'.format(feature_name))
    ft_train_h5f = h5py.File(ft_train_h5_path, 'r')
    ft_val_h5f = h5py.File(ft_val_h5_path, 'r')
    for video in ft_train_h5f.keys():
        ft_dict[video] = ft_train_h5f[video]['fts'][()]
    for video in ft_val_h5f.keys():
        ft_dict[video] = ft_val_h5f[video]['fts'][()]
    
    for cv in tqdm(range(1, 6)):
        cv_train_target_path = os.path.join(resplit_target_dir, 'train_s{}_valid_targets.h5'.format(cv))
        cv_val_target_path = os.path.join(resplit_target_dir, 'val_s{}_valid_targets.h5'.format(cv))
        cv_train_features_path = os.path.join(resplit_features_dir, 'train_s{}_{}.h5'.format(cv, feature_name))
        cv_val_features_path = os.path.join(resplit_features_dir, 'val_s{}_{}.h5'.format(cv, feature_name))
        cv_train_target_h5f = h5py.File(cv_train_target_path, 'r')
        cv_val_target_h5f = h5py.File(cv_val_target_path, 'r')
        cv_train_features_h5f = h5py.File(cv_train_features_path, 'w')
        cv_val_features_h5f = h5py.File(cv_val_features_path, 'w')

        for video in cv_train_target_h5f.keys():
            cv_train_features_h5f.create_group(video)
            cv_train_features_h5f[video]['fts'] = ft_dict[video]
        for video in cv_val_target_h5f.keys():
            cv_val_features_h5f.create_group(video)
            cv_val_features_h5f[video]['fts'] = ft_dict[video]