'''
从ztg抽好的denseface文件转换为适配我这里target的denseface.h5文件
'''

import os
import h5py
import numpy as np
import glob
from tqdm import tqdm

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

dataset_root = '/data9/datasets/Aff-Wild2/'
features_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features'
targets_dir = '/data9/hzp/ABAW_VA_2022/processed_data/targets'
ztg_file = '/data9/datasets/Aff-Wild2/h5_data/VA_Estimation_Challenge/feature/denseface.h5'

ztg_h5f = h5py.File(ztg_file, 'r')
ztg_set_list = ['trn', 'val']
set_list = ['train', 'val']

special_file = os.path.join(targets_dir, 'special_videos.h5')
special_h5f = h5py.File(special_file, 'r')

for ztg_set, set_name in zip(ztg_set_list, set_list):
    print('--------------process {}--------------'.format(set_name))
    valid_targets_path = os.path.join(targets_dir, '{}_valid_targets.h5'.format(set_name))
    valid_targets_h5f = h5py.File(valid_targets_path, 'r')
    denseface_path = os.path.join(features_dir, '{}_denseface.h5'.format(set_name))
    denseface_h5f = h5py.File(denseface_path, 'w')

    for video in tqdm(list(valid_targets_h5f.keys())):
        if valid_targets_h5f[video]['special'][()] == 0:
            video_group = denseface_h5f.create_group(video)
            fts = ztg_h5f[ztg_set][video]['feature'][()]
            assert len(fts) == valid_targets_h5f[video]['length'][()]
            video_group['fts'] = fts
            valid = [0 if not i.any() else 1 for i in fts]
            video_group['valid'] = np.array(valid)

        else: # 后切出来的片段
            original_video = '_'.join(video.split('_')[:-1])
            video_group = denseface_h5f.create_group(video)
            seg_start = special_h5f[original_video][video]['start'][()]
            seg_end = special_h5f[original_video][video]['end'][()]
            whole_video_fts = ztg_h5f[ztg_set][original_video]['feature'][()]
            video_group['fts'] = whole_video_fts[seg_start: seg_end + 1]
            assert len(video_group['fts']) == valid_targets_h5f[video]['length'][()]
            valid = [0 if not i.any() else 1 for i in video_group['fts'][()]]
            video_group['valid'] = np.array(valid)
