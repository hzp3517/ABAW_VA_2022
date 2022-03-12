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

from tools.densenet import DensenetExtractor

gpu_id = 1
pretrained_model_file = 'denseface_run1_1_net_dense_mse1239.pth'

features_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features'
set_list = ['train', 'val']

model_root = '/data8/hzp/ABAW_VA_2022/pretrained_models/'
model_path = os.path.join(model_root, pretrained_model_file)
densenet_extractor = DensenetExtractor(model_path, gpu_id=gpu_id)

for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))

    img_path = os.path.join(features_dir, '{}_gray_img_data.h5'.format(set_name))
    img_h5f = h5py.File(img_path, 'r')
    densenet_path = os.path.join(features_dir, '{}_denseface_mse1239.h5'.format(set_name))
    densenet_h5f = h5py.File(densenet_path, 'w')

    for video in tqdm(list(img_h5f.keys())):
        video_group = densenet_h5f.create_group(video)
        img_list = img_h5f[video]['images'][()]
        pad_list = img_h5f[video]['pad'][()]
        fts = densenet_extractor(img_list, chunk_size=256)
        video_group['fts'] = fts
        video_group['pad'] = pad_list