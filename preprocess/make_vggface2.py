'''
{
    [new_video_id]: 
    {
        'fts': [ft, ft, ...], # (video_len, dim=8631)
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

from tools.vggface2 import Vggface2Extractor

gpu_id = 2
features_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features'
set_list = ['train', 'val']

vggface2_extractor = Vggface2Extractor(gpu_id=gpu_id)

for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))

    img_path = os.path.join(features_dir, '{}_original_img_data.h5'.format(set_name))
    img_h5f = h5py.File(img_path, 'r')
    vggface2_path = os.path.join(features_dir, '{}_vggface2_2048.h5'.format(set_name))
    vggface2_h5f = h5py.File(vggface2_path, 'w')

    for video in tqdm(list(img_h5f.keys())):
        video_group = vggface2_h5f.create_group(video)
        img_list = img_h5f[video]['images'][()]
        pad_list = img_h5f[video]['pad'][()]
        fts = vggface2_extractor(img_list)
        video_group['fts'] = fts
        video_group['pad'] = pad_list