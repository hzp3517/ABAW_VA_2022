'''
{
    [new_video_id]:
    {
        'fts': [ft, ft, ...], # (video_len, dim=342)
        'pad': [0, 0, 1, ...] # 0表示原始图像有效，1表示该帧为从旁边帧填充
    }
}
'''
import os
import h5py
import numpy as np
import glob
from tqdm import tqdm
import cv2
import csv
import pandas as pd

features_dir = '/data2/hzp/ABAW_VA_2022/processed_data/features'
targets_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets'
situ_dir = '/data2/hzp/ABAW_VA_2022/processed_data/situ_fau/'
save_dir = '/data2/hzp/ABAW_VA_2022/processed_data/features/'

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

set_list = ['train', 'val']

special_file = os.path.join(targets_dir, 'special_videos.h5')
special_h5f = h5py.File(special_file, 'r')
special_videos = list(special_h5f.keys())


def get_fau_ft(video_id, video_frame_ids):
    '''
    video_frame_ids: [00001, ...]
    
    return:
    - video_ft
    - pad: [0, 0, 1, ...] # 0表示使用原始图像特征，1表示使用邻近特征填充
    '''
    valid_len_video = len(video_frame_ids)
    pad = []

    video_ft = []
    for idx in range(len(video_frame_ids)):
        frame_id = video_frame_ids[idx]
        if frame_id in situ_dict[video_id].keys():
            pad.append(0)
            video_ft.append(situ_dict[video_id][frame_id])
        else:
            # 特征不存在，寻找附近帧
            delta = 1
            while 1:
                if idx - delta > 0:
                    former_frame_id = video_frame_ids[idx-delta]
                else:
                    former_frame_id = None
                    
                if former_frame_id in situ_dict[video_id].keys():
                    pad.append(1)
                    video_ft.append(situ_dict[video_id][former_frame_id])
                    break

                if idx + delta < len(video_frame_ids):
                    later_frame_id = video_frame_ids[idx-delta]
                else:
                    later_frame_id = None

                if later_frame_id in situ_dict[video_id].keys():
                    pad.append(1)
                    video_ft.append(situ_dict[video_id][later_frame_id])
                    break
                
                delta += 1

    assert len(video_ft) == valid_len_video
    assert len(pad) == valid_len_video
    return video_ft, pad


for set_name in set_list:
    print('------------process set-{}------------------'.format(set_name))
    original_targets_path = os.path.join(targets_dir, '{}_original_all_targets.h5'.format(set_name))
    valid_targets_path = os.path.join(targets_dir, '{}_valid_targets.h5'.format(set_name))
    ft_path = os.path.join(save_dir, '{}_FAU_situ.h5'.format(set_name))
    original_h5f = h5py.File(original_targets_path, 'r')
    valid_h5f = h5py.File(valid_targets_path, 'r')
    ft_h5f = h5py.File(ft_path, 'w')
    csv_path = os.path.join(situ_dir, set_name+'.csv')
    npy_path = os.path.join(situ_dir, set_name+'.npy')
    csv_data = pd.read_csv(csv_path)
    npy_data = np.load(npy_path)
    embedd_idx = csv_data['embedd_idx']
    situ_dict = {}

    for ebd_idx, ft in tqdm(list(zip(embedd_idx, npy_data))):
        video_id, frame_id = ebd_idx.strip().split('/')
        if video_id not in situ_dict.keys():
            situ_dict[video_id] = {}
        situ_dict[video_id][frame_id] = ft

    for video in tqdm(list(original_h5f.keys())):
        if video not in special_videos:
            video_group = ft_h5f.create_group(video)
            video_len = valid_h5f[video]['length'][()]
            video_frame_ids = [str(i).zfill(5) for i in range(1, video_len + 1)]
            video_ft, pad = get_fau_ft(video, video_frame_ids)
            video_group['fts'] = np.array(video_ft).astype(np.float32)
            assert len(video_group['fts']) == video_len
            video_group['pad'] = np.array(pad)
        else:
            for new_video in special_h5f[video].keys():
                start_idx = special_h5f[video][new_video]['start'][()]
                end_idx = special_h5f[video][new_video]['end'][()] + 1
                video_group = ft_h5f.create_group(new_video)
                video_frame_ids = [str(i).zfill(5) for i in range(start_idx + 1, end_idx + 1)]
                video_ft, pad = get_fau_ft(video, video_frame_ids)
                video_group['fts'] = np.array(video_ft).astype(np.float32)
                assert len(video_group['fts']) == special_h5f[video][new_video]['length'][()]
                video_group['pad'] = np.array(pad)




