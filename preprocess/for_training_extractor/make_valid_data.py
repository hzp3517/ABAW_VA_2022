'''
将所有有效图像按视频存，每个视频存成一个大数组
[set]_imgs.h5
{
    [video_id]: [img, img, ...] # 存未处理过的图像（112*112*3）
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

new_target_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets/for_training_extractor/'
img_root = '/data2/hzp/Aff-Wild2/cropped_aligned/'
save_dir = '/data2/hzp/ABAW_VA_2022/processed_data/features/for_training_extractor/'
set_list = ['train', 'val']

mkdir(save_dir)

for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))
    target_path = os.path.join(new_target_dir, '{}_targets.h5'.format(set_name))
    target_h5f = h5py.File(target_path, 'r')
    save_path = os.path.join(save_dir, '{}_imgs.h5'.format(set_name))
    save_h5f = h5py.File(save_path, 'w')

    for video in tqdm(list(target_h5f.keys())):
        img_list = []
        frame_list = target_h5f[video]['frame'][()]
        frame_list = [os.path.join(img_root, video, str(i).zfill(5) + '.jpg') for i in frame_list]

        for frame in frame_list:
            img = cv2.imread(frame)
            img_list.append(img)
        imgs = np.stack(img_list).astype(np.uint8)
        save_h5f[video] = imgs