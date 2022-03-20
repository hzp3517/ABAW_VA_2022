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

import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code/preprocess')
from tools.affectnet import AffectnetExtractor

gpu_id = 1
set_name = 'test'

features_dir = '/data2/hzp/ABAW_VA_2022/processed_data/features'
# set_list = ['train', 'val']

affectnet_extractor = AffectnetExtractor(gpu_id=gpu_id)

img_path = os.path.join(features_dir, '{}_gray_img_data.h5'.format(set_name))
img_h5f = h5py.File(img_path, 'r')
affectnet_path = os.path.join(features_dir, '{}_affectnet.h5'.format(set_name))
affectnet_h5f = h5py.File(affectnet_path, 'w')

for video in tqdm(list(img_h5f.keys())):
    video_group = affectnet_h5f.create_group(video)
    img_list = img_h5f[video]['images'][()]
    pad_list = img_h5f[video]['pad'][()]
    fts = affectnet_extractor(img_list, chunk_size=256)
    video_group['fts'] = fts
    video_group['pad'] = pad_list