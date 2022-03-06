'''
have some problem now!
'''

import h5py
import numpy as np
import torch
from tqdm import tqdm
import os
import glob
import cv2
import math


def cal_mean_std(h5f_list, save_dir): #注意：颜色通道顺序为BGR以适应opencv
    '''
    方差计算公式：s² = (x1² + x2²+ ... + xn²)/n - [(x1 + x2 + ... + xn)/n]²，即：(平方平均)²-(算数平均)²，也即：平方和的均值-(算数平均)²
    标准差为 s
    '''
    # img_size = 224
    mean_list = [] #每个视频中所有像素的算术平均
    pow_mean_list = [] #每个视频中所有像素的平方平均
    img_num_list = []

    for h5f in h5f_list:
        video_ids = h5f.keys()
        for video_id in tqdm(video_ids):
            imgs = h5f[video_id]['images'][()] # (len, 224, 224, 3)
            cnt_img = len(imgs)
            imgs = imgs.reshape(-1, 3)
            imgs = imgs / 255.0 #将像素值的范围缩到[0, 1]

            pow_imgs = pow(imgs, 2)
            video_mean = imgs.mean(axis=0)
            pow_video_mean = pow_imgs.mean(axis=0)

            mean_list.append(video_mean)
            pow_mean_list.append(pow_video_mean)
            img_num_list.append(cnt_img)
    
    mean_array = np.stack(mean_list)
    pow_mean_array = np.stack(pow_mean_list)
    img_num_array = np.stack(img_num_list)

    all_img_num = img_num_array.sum()

    img_num_array = np.expand_dims(img_num_array, axis=1) #扩充维度
    img_num_array = np.repeat(img_num_array, 3, axis=1) #在对应维度重复

    weighted_mean_array = mean_array * img_num_array # *为对应位置相乘
    weighted_pow_mean_array = pow_mean_array * img_num_array
    # global_mean_array = weighted_mean_array / all_img_num
    # global_pow_mean_array = weighted_pow_mean_array / all_img_num

    mean = weighted_mean_array.sum(axis=0) / all_img_num
    pow_mean = weighted_pow_mean_array.sum(axis=0) / all_img_num
    var = pow_mean - pow(mean, 2)
    std = np.array([math.sqrt(i) for i in var])

    print('MEAN:', mean)
    print('STD:', std)

    # 记录下运算的结果：
    record_file = os.path.join(save_dir, 'mean_std.txt')
    with open(record_file, 'w') as f:
        f.write('mean:' + str(mean) + '\n')
        f.write('std:' + str(std) + '\n')


if __name__ == '__main__':
    features_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features/'
    set_list = ['train', 'val']
    h5f_list = []
    for set_name in set_list:
        original_img_path = os.path.join(features_dir, '{}_original_img_data.h5'.format(set_name))
        original_img_h5f = h5py.File(original_img_path, 'r')
        h5f_list.append(original_img_h5f)
    save_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features/' #计算出的mean和std的结果文件保存目录
    cal_mean_std(h5f_list, save_dir)
