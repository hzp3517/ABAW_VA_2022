import os
import os.path as osp
import shutil
import h5py
import numpy as np
import cv2
from tqdm import tqdm
import math
import json
import glob

def cal_mean_std(gray_img_h5f, save_dir):
    img_size = 64
    all_faces = []
    
    video_ids = gray_img_h5f.keys()

    for video_id in tqdm(video_ids):
        imgs = gray_img_h5f[video_id]['images'][()] # (len, 64, 64)
        imgs = imgs.reshape(-1)
        all_faces.append(imgs)

    all_faces = np.concatenate(all_faces)
    print(all_faces.shape)

    mean = all_faces.mean()
    std = all_faces.std()

    print('MEAN:', mean)
    print('STD:', std)

    #记录下运算的结果：
    record_file = os.path.join(save_dir, 'gray_face_mean_std.txt')
    with open(record_file, 'w') as f:
        f.write('mean:' + str(mean) + '\n')
        f.write('std:' + str(std) + '\n')


if __name__ == '__main__':
    features_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features/'
    # set_list = ['train'] # 只计算训练集的mean和std
    # h5f_list = []
    # for set_name in set_list:
    gray_img_path = os.path.join(features_dir, 'train_gray_img_data.h5')
    gray_img_h5f = h5py.File(gray_img_path, 'r')
    save_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features/' #计算出的mean和std的结果文件保存目录
    cal_mean_std(gray_img_h5f, save_dir)