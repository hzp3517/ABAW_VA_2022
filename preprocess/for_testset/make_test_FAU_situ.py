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

set_name = 'test'

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

features_dir = '/data2/hzp/ABAW_VA_2022/processed_data/features'
affectnet_path = os.path.join(features_dir, 'test_affectnet.h5')
affectnet_h5f = h5py.File(affectnet_path, 'r')
video_id_lst = list(affectnet_h5f.keys())
target = {}
for video in video_id_lst:
    target[video] = len(affectnet_h5f[video]['fts'][()])

situ_dir = '/data2/hzp/ABAW_VA_2022/processed_data/situ_fau/'
ft_path = os.path.join(features_dir, 'test_FAU_situ.h5')
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
        assert video_id in video_id_lst
        situ_dict[video_id] = {}
    situ_dict[video_id][frame_id] = ft

for video in tqdm(video_id_lst):
    video_group = ft_h5f.create_group(video)
    video_len = target[video]
    video_frame_ids = [str(i).zfill(5) for i in range(1, video_len + 1)]
    video_ft, pad = get_fau_ft(video, video_frame_ids)
    video_group['fts'] = np.array(video_ft).astype(np.float32)
    assert len(video_group['fts']) == video_len
    video_group['pad'] = np.array(pad)