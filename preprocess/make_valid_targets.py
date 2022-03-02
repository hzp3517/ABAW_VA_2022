'''
把所有视频的所有标注都存到一个h5文件中
{
    [video_id]: {
        'valence': [0.1, 0.1s, ...],
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
special_file = os.path.join(target_dir, 'special_videos.h5')
set_list = ['train', 'val']

for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))
    origin_targets_path = os.path.join(target_dir, '{}_original_all_targets.h5'.format(set_name))
    origin_targets_h5f = h5py.File(origin_targets_path, 'r')
    special_h5f = h5py.File(special_file, 'r')
    valid_targets_path = os.path.join(target_dir, '{}_valid_targets.h5'.format(set_name))
    valid_targets_h5f = h5py.File(valid_targets_path, 'w')

    problem_videos = list(special_h5f.keys())
    for video in origin_targets_h5f.keys():
        if video not in problem_videos:
            video_group = valid_targets_h5f.create_group(video)
            video_group['valence'] = origin_targets_h5f[video]['valence'][()]
            video_group['arousal'] = origin_targets_h5f[video]['arousal'][()]
            video_group['special'] = 0
            video_group['length'] = origin_targets_h5f[video]['length'][()]
        else:
            segment_list = list(special_h5f[video].keys())
            for seg in segment_list:
                video_group = valid_targets_h5f.create_group(seg)
                seg_start = special_h5f[video][seg]['start'][()]
                seg_end = special_h5f[video][seg]['end'][()]
                video_group['valence'] = origin_targets_h5f[video]['valence'][()][seg_start: seg_end + 1]
                video_group['arousal'] = origin_targets_h5f[video]['arousal'][()][seg_start: seg_end + 1]
                video_group['special'] = 1
                video_group['length'] = special_h5f[video][seg]['length'][()]