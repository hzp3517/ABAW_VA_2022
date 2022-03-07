'''
[set]_224_color_img.h5
{
    [frame_id]: img (224*224*3)
}

[set]_64_gray_img.h5
{
    [frame_id]: img (64*64)
}
'''
import os
import h5py
import numpy as np
import glob
from tqdm import tqdm
import cv2

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

def convert_img(img, img_size=64):
    '''
    return:
    - success
    - img
    '''
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_size, img_size))
    return img

origin_target_dir = '/data8/hzp/ABAW_VA_2022/processed_data/features/'
save_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features/for_training_extractor'
mkdir(save_dir)

set_list = ['train', 'val']

for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))
    ori_img_path = os.path.join(origin_target_dir, '{}_frame_img_data.h5'.format(set_name))
    ori_img_h5f = h5py.File(ori_img_path, 'r')
    save_color_path = os.path.join(save_dir, '{}_224_color_img.h5'.format(set_name))
    save_color_h5f = h5py.File(save_color_path, 'w')
    save_gray_path = os.path.join(save_dir, '{}_64_gray_img.h5'.format(set_name))
    save_gray_h5f = h5py.File(save_gray_path, 'w')
    
    for image in tqdm(list(ori_img_h5f.keys())):
        if ori_img_h5f[image]['pad'] == 0: # 原始图像存在
            color_img = ori_img_h5f[image]['image'][()]
            save_color_h5f[image] = color_img
            gray_img = convert_img(color_img)
            save_gray_h5f[image] = gray_img
            print(gray_img.shape)
            
        break
            