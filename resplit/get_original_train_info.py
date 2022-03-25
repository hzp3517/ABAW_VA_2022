'''
resplit/original_train_info.h5
[original_video_id]:
{
    ['frame_nb']: 帧数
    ['valence']: [x, y, z]（分别对应[-1, 0), [0, 0.4), [0.4, 1]区间的数量）
    ['arousal']: [x, y, z]（分别对应[-1, 0), [0, 0.4), [0.4, 1]区间的数量）
}
'''

import h5py
import numpy as np
import os
from tqdm import tqdm

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

targets_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets/'
save_dir = os.path.join(targets_dir, 'resplit')
mkdir(save_dir)
info_path = os.path.join(save_dir, 'original_train_info.h5')
info_h5f = h5py.File(info_path, 'w')
original_target_path = os.path.join(targets_dir, 'train_original_all_targets.h5')
special_target_path = os.path.join(targets_dir, 'special_videos.h5')
valid_target_path = os.path.join(targets_dir, 'train_valid_targets.h5')
original_h5f = h5py.File(original_target_path, 'r')
special_h5f = h5py.File(special_target_path, 'r')
valid_h5f = h5py.File(valid_target_path, 'r')
speicial_videos = list(special_h5f.keys())


def get_section(labels):
    sec_1 = labels < 0
    sec_3 = labels >= 0.4
    len_sec_2 = len(labels) - len(labels[sec_1]) - len(labels[sec_3])
    return len(labels[sec_1]), len_sec_2, len(labels[sec_3])


for video in tqdm(list(original_h5f.keys())):
    video_group = info_h5f.create_group(video)
    if video not in speicial_videos:
        video_group['frame_nb'] = original_h5f[video]['length'][()]
        valence_section = list(get_section(original_h5f[video]['valence'][()]))
        arousal_section = list(get_section(original_h5f[video]['arousal'][()]))
        video_group['valence'] = np.array(valence_section)
        video_group['arousal'] = np.array(arousal_section)
    else:
        frame_nb = 0
        valence_sec = [0, 0, 0]
        arousal_sec = [0, 0, 0]
        for new_video in special_h5f[video].keys():
            frame_nb += valid_h5f[new_video]['length'][()]
            valence_section_tmp = list(get_section(valid_h5f[new_video]['valence'][()]))
            arousal_section_tmp = list(get_section(valid_h5f[new_video]['arousal'][()]))
            for i in range(3):
                valence_sec[i] += valence_section_tmp[i]
                arousal_sec[i] += arousal_section_tmp[i]
        video_group['frame_nb'] = frame_nb
        video_group['valence'] = np.array(valence_sec)
        video_group['arousal'] = np.array(arousal_sec)

