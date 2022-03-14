'''
利用openface对所有视频cropped人脸图像抽取low-level特征，保存成csv文件
'''
import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv
import multiprocessing
import glob
from tools.openface import OpenFaceLandmarkImg

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

# def get_openface_csv(face_root, save_root):
#     openface = OpenFaceTracker(save_root=save_root)
#     ori_video_ids = [get_basename(i) for i in  os.listdir(face_root)]
#     for ori_video_id in tqdm(ori_video_ids):
#         face_dir = osp.join(face_root, ori_video_id)
#         save_dir = openface(face_dir) #对应video_id的csv保存路径

def get_openface_csv(face_dir):
    save_dir = openface(face_dir) #对应video_id的csv保存路径


if __name__ == '__main__':
    dataset_face_root = '/data9/datasets/Aff-Wild2/cropped_images/'
    csv_save_root = '/data9/hzp/ABAW_VA_2022/processed_data/openface_save'
    mkdir(csv_save_root)

    # path_list = []
    face_dir_list = []
    openface = OpenFaceLandmarkImg(save_root=csv_save_root)
    pool = multiprocessing.Pool(16) #python进程池
    ori_video_ids = [get_basename(i) for i in os.listdir(dataset_face_root)]

    for ori_video_id in ori_video_ids:
        face_dir = os.path.join(dataset_face_root, ori_video_id)
        # save_dir = os.path.join(csv_save_root, ori_video_id)
        # path_list.append((face_dir, save_dir))
        face_dir_list.append(face_dir)
    
    # list(tqdm(pool.imap(get_openface_csv, path_list), total=len(path_list)))
    list(tqdm(pool.imap(get_openface_csv, face_dir_list), total=len(face_dir_list)))

    pool.close()
    pool.join()