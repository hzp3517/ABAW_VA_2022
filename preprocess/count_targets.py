'''
统计一下缺失有效标注的帧占比，以及含有缺失标注的视频占比
'''
import os
import h5py
import numpy as np
import glob
from tqdm import tqdm

dataset_root = '/data9/datasets/Aff-Wild2/'
save_dir = '/data9/hzp/ABAW_VA_2022/processed_data/targets'
set_list = ['train', 'val']

# for set_name in set_list:
#     print('--------------process {}--------------'.format(set_name))
#     all_targets_path = os.path.join(save_dir, '{}_original_all_targets.h5'.format(set_name))
#     all_targets_h5f = h5py.File(all_targets_path, 'r')

#     cnt_good_videos = 0
#     cnt_valid_frames = 0
#     cnt_frames = 0
#     num_videos = len(all_targets_h5f.keys())

#     for video in tqdm(all_targets_h5f.keys()):
#         if np.array(all_targets_h5f[video]['valid']).all():
#             cnt_good_videos += 1
#         else:
#             print(video)
#         for frame in all_targets_h5f[video]['valid']:
#             cnt_frames += 1
#             if frame == 1:
#                 cnt_valid_frames += 1
    
#     print('good_videos: {}\tratio: {}'.format(cnt_good_videos, cnt_good_videos * 1.0 / num_videos))
#     print('invalid_frames: {}\tratio: {}'.format(cnt_frames - cnt_valid_frames, cnt_valid_frames * 1.0 / cnt_frames))



trn_problem_videos = ['10-60-1280x720_right', '86-24-1920x1080']
val_problem_video = 'video59_right'

for video in trn_problem_videos:
    print('-------------video {}---------------'.format(video))
    all_targets_path = os.path.join(save_dir, '{}_original_all_targets.h5'.format('train'))
    all_targets_h5f = h5py.File(all_targets_path, 'r')
    valid = all_targets_h5f[video]['valid']
    cnt_seg = 0
    cnt_len = 0
    
    i = 0
    while i < len(valid):
        if valid[i] != 0:
            cnt_len += 1
            i += 1
        elif i == 0:
            while i < len(valid):
                if valid[i] != 0:
                    cnt_len += 1
                    i += 1
                    break
                i += 1
        else:
            print('seg: {}'.format(cnt_len))
            cnt_len = 0
            cnt_seg += 1
            while i < len(valid):
                if valid[i] != 0:
                    cnt_len += 1
                    i += 1
                    break
                i += 1
    if valid[-1] != 0:
        print('seg: {}'.format(cnt_len))

video = val_problem_video
print('-------------video {}---------------'.format(video))
all_targets_path = os.path.join(save_dir, '{}_original_all_targets.h5'.format('val'))
all_targets_h5f = h5py.File(all_targets_path, 'r')
valid = all_targets_h5f[video]['valid']
cnt_seg = 0
cnt_len = 0

i = 0
while i < len(valid):
    if valid[i] != 0:
        cnt_len += 1
        i += 1
    elif i == 0:
        while i < len(valid):
            if valid[i] != 0:
                cnt_len += 1
                i += 1
                break
            i += 1
    else:
        print('seg: {}'.format(cnt_len))
        cnt_len = 0
        cnt_seg += 1
        while i < len(valid):
            if valid[i] != 0:
                cnt_len += 1
                i += 1
                break
            i += 1
if valid[-1] != 0:
    print('seg: {}'.format(cnt_len))


