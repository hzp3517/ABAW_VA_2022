'''
把所有视频的所有标注都存到一个h5文件中
{
    [video_id]: {
        'valence': [0.1, 0.1, ...],
        'arousal': [0.2, 0.2, ...],
        'special': 0或1, # 1表示这一段是切出来之后的片段，0表示这个视频就是原本对应的id
        'length': xx
    }
}
'''
import os
import h5py
import numpy as np
import glob
from tqdm import tqdm
import json

target_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets'
resplit_dir = os.path.join(target_dir, 'resplit')
special_file = os.path.join(target_dir, 'special_videos.h5')
ori_train_file = os.path.join(target_dir, 'train_valid_targets.h5')
ori_val_file = os.path.join(target_dir, 'val_valid_targets.h5')
special_h5f = h5py.File(special_file, 'r')
special_videos = list(special_h5f.keys())
ori_train_h5f = h5py.File(ori_train_file, 'r')
ori_val_h5f = h5py.File(ori_val_file, 'r')
spilt_file = os.path.join(resplit_dir, 'train_split.json')
with open(spilt_file, 'r') as f:
    resplit_dict = json.load(f)

for cv in tqdm(range(1, 6)):
    cv_train_target_path = os.path.join(resplit_dir, 'train_s{}_valid_targets.h5'.format(cv))
    cv_val_target_path = os.path.join(resplit_dir, 'val_s{}_valid_targets.h5'.format(cv))
    cv_train_h5f = h5py.File(cv_train_target_path, 'w')
    cv_val_h5f = h5py.File(cv_val_target_path, 'w')

    for ite_cv in range(1, 6):
        if ite_cv == cv: # 作为新验证集
            for video in resplit_dict['split_{}'.format(ite_cv)]:
                if video not in special_videos:
                    cv_val_h5f.create_group(video)
                    cv_val_h5f[video]['valence'] = ori_train_h5f[video]['valence'][()]
                    cv_val_h5f[video]['arousal'] = ori_train_h5f[video]['arousal'][()]
                    cv_val_h5f[video]['special'] = ori_train_h5f[video]['special'][()]
                    cv_val_h5f[video]['length'] = ori_train_h5f[video]['length'][()]
                else:
                    for new_video in special_h5f[video].keys():
                        cv_val_h5f.create_group(new_video)
                        cv_val_h5f[new_video]['valence'] = ori_train_h5f[new_video]['valence'][()]
                        cv_val_h5f[new_video]['arousal'] = ori_train_h5f[new_video]['arousal'][()]
                        cv_val_h5f[new_video]['special'] = ori_train_h5f[new_video]['special'][()]
                        cv_val_h5f[new_video]['length'] = ori_train_h5f[new_video]['length'][()]
        
        else:
            for video in resplit_dict['split_{}'.format(ite_cv)]:
                if video not in special_videos:
                    cv_train_h5f.create_group(video)
                    cv_train_h5f[video]['valence'] = ori_train_h5f[video]['valence'][()]
                    cv_train_h5f[video]['arousal'] = ori_train_h5f[video]['arousal'][()]
                    cv_train_h5f[video]['special'] = ori_train_h5f[video]['special'][()]
                    cv_train_h5f[video]['length'] = ori_train_h5f[video]['length'][()]
                else:
                    for new_video in special_h5f[video].keys():
                        cv_train_h5f.create_group(new_video)
                        cv_train_h5f[new_video]['valence'] = ori_train_h5f[new_video]['valence'][()]
                        cv_train_h5f[new_video]['arousal'] = ori_train_h5f[new_video]['arousal'][()]
                        cv_train_h5f[new_video]['special'] = ori_train_h5f[new_video]['special'][()]
                        cv_train_h5f[new_video]['length'] = ori_train_h5f[new_video]['length'][()]

    for video in ori_val_h5f.keys():
        if video not in special_videos:
            cv_train_h5f.create_group(video)
            cv_train_h5f[video]['valence'] = ori_val_h5f[video]['valence'][()]
            cv_train_h5f[video]['arousal'] = ori_val_h5f[video]['arousal'][()]
            cv_train_h5f[video]['special'] = ori_val_h5f[video]['special'][()]
            cv_train_h5f[video]['length'] = ori_val_h5f[video]['length'][()]
        else:
            for new_video in special_h5f[video].keys():
                cv_train_h5f.create_group(new_video)
                cv_train_h5f[new_video]['valence'] = ori_val_h5f[new_video]['valence'][()]
                cv_train_h5f[new_video]['arousal'] = ori_val_h5f[new_video]['arousal'][()]
                cv_train_h5f[new_video]['special'] = ori_val_h5f[new_video]['special'][()]
                cv_train_h5f[new_video]['length'] = ori_val_h5f[new_video]['length'][()]
                


