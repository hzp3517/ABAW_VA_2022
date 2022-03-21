'''
delete preds of invalid frames and split the video
'''
import h5py
import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys

target_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets'
origin_targets_path = os.path.join(target_dir, 'val_original_all_targets.h5')
special_file = os.path.join(target_dir, 'special_videos.h5')
situ_dir = '/data2/hzp/ABAW_VA_2022/situ_pred_val_0321/'

origin_targets_h5f = h5py.File(origin_targets_path, 'r')
special_h5f = h5py.File(special_file, 'r')
special_lst = list(special_h5f.keys())

for target in ['arousal', 'valence']:
    print('--------------process {}--------------'.format(target))
    new_pred_dict = {}

    situ_path = os.path.join(situ_dir, '{}_situ_0321.json'.format(target))
    with open(situ_path, 'r') as f:
        pred_dict = json.load(f)

    for video in tqdm(pred_dict.keys()):
        if video in special_lst:
            new_video_id_lst = list(special_h5f[video].keys())
            for new_id in new_video_id_lst:
                start = special_h5f[video][new_id]['start'][()]
                end = special_h5f[video][new_id]['end'][()]
                new_pred_dict[new_id] = pred_dict[video][start: end + 1]
        else:
            new_pred_dict[video] = pred_dict[video]

    save_path = os.path.join(situ_dir, 'new_{}_situ_0321.json'.format(target))
    json.dump(new_pred_dict, open(save_path, 'w'))
