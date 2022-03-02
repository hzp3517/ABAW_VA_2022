'''
{
    [frame_id]: { # frame_id: [new_video_id]_[frame_idx]
        'valence': 0.1,
        'arousal': 0.2
    }
}
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

target_dir = '/data9/hzp/ABAW_VA_2022/processed_data/targets'
set_list = ['train', 'val']

for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))
    valid_targets_path = os.path.join(target_dir, '{}_valid_targets.h5'.format(set_name))
    valid_targets_h5f = h5py.File(valid_targets_path, 'r')
    frame_targets_path = os.path.join(target_dir, '{}_frame_targets.h5'.format(set_name))
    frame_targets_h5f = h5py.File(frame_targets_path, 'w')

    for video in tqdm(valid_targets_h5f.keys()):
        valence_list = valid_targets_h5f[video]['valence'][()]
        arousal_list = valid_targets_h5f[video]['arousal'][()]
        v_len = valid_targets_h5f[video]['length'][()]
        
        for idx in range(1, v_len + 1):
            frame_id = str(idx).zfill(5)
            frame_id = video + '_' + frame_id
            frame_group = frame_targets_h5f.create_group(frame_id)
            frame_group['valence'] = valence_list[idx - 1]
            frame_group['arousal'] = arousal_list[idx - 1]