import os, glob
import torch
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd

import sys
sys.path.append('/data8/hzp/ABAW_VA_2022/code/preprocess')#
from tools.base_worker import BaseWorker

def get_basename(path):
    basename = os.path.basename(path)
    if os.path.isfile(path):
        basename = basename[:basename.rfind('.')]
    return basename

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


# class OpenFaceLandmarkImg(BaseWorker):
#     '''
#     使用openface工具逐帧地抽取人脸信息
#     '''
#     def __init__(self, save_root='/data12/MUSE2021/preprocess/tools/.tmp/openface_save/',
#             openface_dir='/root/tools/OpenFace/build/bin', logger=None): #leo_hzp
#         super().__init__(logger=logger)
#         self.save_root = save_root
#         self.openface_dir = openface_dir

#     def __call__(self, frames_dir, face_id):
#         basename = get_basename(frames_dir)
#         face_file = '{}.jpg'.format(face_id)
#         save_dir = os.path.join(self.save_root, basename, face_id)
#         mkdir(save_dir)
#         cmd = '{}/FaceLandmarkImg -root {} -f {} -mask -out_dir {} > /dev/null 2>&1'.format(
#                     self.openface_dir, frames_dir, face_file, save_dir
#                 )
#         os.system(cmd)
#         return save_dir
    

class OpenFaceLandmarkImg(BaseWorker):
    '''
    使用openface工具，送入一个video的所有帧图像的目录，逐帧地抽取人脸信息。
    生成的目录中，有face_id.csv文件
    '''
    def __init__(self, save_root='/data12/MUSE2021/preprocess/tools/.tmp/openface_save/',
            openface_dir='/root/tools/OpenFace/build/bin', logger=None): #leo_hzp
        super().__init__(logger=logger)
        self.save_root = save_root
        self.openface_dir = openface_dir

    def __call__(self, frames_dir):
        basename = get_basename(frames_dir)
        # face_file = '{}.jpg'.format(face_id)
        save_dir = os.path.join(self.save_root, basename)
        mkdir(save_dir)
        cmd = '{}/FaceLandmarkImg -fdir {} -mask -out_dir {} > /dev/null 2>&1'.format(
                    self.openface_dir, frames_dir, save_dir
                )
        os.system(cmd)
        return save_dir





class OpenFaceTracker(BaseWorker):
    ''' 使用openface工具抽取人脸
        eg: 输入视频帧位置: "/data12/lrc/MUSE2021/data/raw-data-ulm-tsst/data/raw/faces/1"（注意最后不要加斜杠）
            输出人脸图片位置：save_root = '/data12/MUSE2021/preprocess/tools/.tmp/openface_save/1'
            其中openface_save/1/1.csv文件是我们需要的，包含人脸关键点和AU等信息
    '''
    def __init__(self, save_root='/data12/MUSE2021/preprocess/tools/.tmp/openface_save/',
            openface_dir='/root/tools/OpenFace/build/bin', logger=None): #leo_hzp
        super().__init__(logger=logger)
        self.save_root = save_root
        self.openface_dir = openface_dir
    
    def __call__(self, frames_dir):
        basename = get_basename(frames_dir)
        save_dir = os.path.join(self.save_root, basename)
        mkdir(save_dir)
        cmd = '{}/FeatureExtraction -fdir {} -mask -out_dir {} > /dev/null 2>&1'.format(
                    self.openface_dir, frames_dir, save_dir
                )
        os.system(cmd)
        return save_dir



if __name__ == '__main__':
    # face_dir = '/data9/datasets/Aff-Wild2/cropped_images/113'
    # openface = OpenFaceTracker(save_root='/data9/hzp/ABAW_VA_2022/processed_data/openface_save_tmp/')
    # save_dir = openface(face_dir)

    face_dir = '/data9/datasets/Aff-Wild2/cropped_aligned/112'
    # face_id = '00072'
    openface = OpenFaceLandmarkImg(save_root='/data9/hzp/ABAW_VA_2022/processed_data/openface_save_tmp/')
    # save_dir = openface(face_dir, face_id)
    save_dir = openface(face_dir)