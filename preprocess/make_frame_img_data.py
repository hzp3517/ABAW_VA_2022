'''
{
    [frame_id]: 
    {
        'images': img (224*224*3),
        'pad': 0或1 # 0表示原始图像有效，1表示该帧为从旁边帧填充
    }
}
'''
import os
import h5py
import numpy as np
import glob
from tqdm import tqdm
import cv2

features_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features'
set_list = ['train', 'val']

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

for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))
    original_img_path = os.path.join(features_dir, '{}_original_img_data.h5'.format(set_name))
    original_img_h5f = h5py.File(original_img_path, 'r')
    frame_img_path = os.path.join(features_dir, '{}_frame_img_data.h5'.format(set_name))
    frame_img_h5f = h5py.File(frame_img_path, 'w')

    for video in tqdm(original_img_h5f.keys()):
        img_list = original_img_h5f[video]['images'][()]
        pad_list = original_img_h5f[video]['pad'][()]
        v_len = len(pad_list)

        for idx in range(1, v_len + 1):
            frame_id = str(idx).zfill(5)
            frame_id = video + '_' + frame_id
            frame_group = frame_img_h5f.create_group(frame_id)
            frame_group['image'] = img_list[idx - 1]
            frame_group['pad'] = pad_list[idx - 1]