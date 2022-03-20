import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv
import math

import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code/preprocess')
from tools.vggish import VggishExtractor

set_name = 'test'

def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError:
        pass


def get_vggish_ft(extractor, audio_root, video_id, num_frames, gpu_id):
    vggish = extractor # vggish = VggishExtractor(seg_len=1/30, step_size=1/30, device=gpu_id) #标签timestamp是30Hz
    audio_path = osp.join(audio_root, video_id + '.wav')
    ft = vggish(audio_path)
    if ft.shape[0] >= num_frames:
        ft = ft[:num_frames]
    else:
        pad_ft = []
        for i in range(num_frames - ft.shape[0]):
            pad_ft.append(ft[-1])
        pad_ft = np.stack(pad_ft)
        ft = np.concatenate((ft, pad_ft), axis=0)
    ft = ft.astype(np.float32)
    return ft


def make_vggish_feature(target_dir, audio_root, save_dir, gpu_id):
    extractor = VggishExtractor(seg_len=1/30, step_size=1/30, device=gpu_id) #标签timestamp是30Hz

    # special_targets_path = os.path.join(target_dir, 'special_videos.h5')
    # special_h5f = h5py.File(special_targets_path, 'r')


    # original_targets_path = os.path.join(target_dir, '{}_original_all_targets.h5'.format(set_name))
    valid_targets_path = os.path.join(target_dir, '{}_valid_targets.h5'.format(set_name))
    ft_path = os.path.join(save_dir, '{}_vggish.h5'.format(set_name))
    # original_h5f = h5py.File(original_targets_path, 'r')
    valid_h5f = h5py.File(valid_targets_path, 'r')
    ft_h5f = h5py.File(ft_path, 'w')

    valid_video_list = list(valid_h5f.keys())
    for new_video_id in tqdm(valid_video_list):
        video_group = ft_h5f.create_group(new_video_id)
        
        # if valid_h5f[new_video_id]['special'][()] == 0: # 没有被切
        num_frames = valid_h5f[new_video_id]['length'][()]
        video_ft = get_vggish_ft(extractor, audio_root, new_video_id, num_frames, gpu_id)
        video_group['fts'] = video_ft
        # else: # 后切出来的片段
        #     original_video = '_'.join(new_video_id.split('_')[:-1])
        #     num_frames = original_h5f[original_video]['length'][()]
        #     video_ft = get_vggish_ft(extractor, audio_root, original_video, num_frames, gpu_id)
        #     seg_start = special_h5f[original_video][new_video_id]['start'][()]
        #     seg_end = special_h5f[original_video][new_video_id]['end'][()]
        #     video_group['fts'] = video_ft[seg_start: seg_end + 1]
        #     assert len(video_group['fts']) == valid_h5f[new_video_id]['length'][()]


if __name__ == '__main__':
    audio_root = '/data2/hzp/ABAW_VA_2022/processed_data/audios/'
    target_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets/'
    save_dir = '/data2/hzp/ABAW_VA_2022/processed_data/features/'
    mkdir(save_dir)

    print('making vggish')
    make_vggish_feature(target_dir, audio_root, save_dir, gpu_id=7)