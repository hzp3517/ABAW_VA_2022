'''
检查一下训练集和验证集所有视频的帧数，验证读视频获取帧数的方法是否有问题
'''
import os
import h5py
import numpy as np
import glob
from tqdm import tqdm
import ffmpeg
import cv2

def get_nb_frames_opencv(video_path):
    video_cap = cv2.VideoCapture(video_path)
    frame_count = 0
    # all_frames = []
    while(True):
        ret, frame = video_cap.read()
        if ret is False:
            break
        frame_count = frame_count + 1
    return frame_count + 1 # 经过验证，发现官方给的帧数总是比检测到的数目多1


def get_nb_frames(video_path):
    info = ffmpeg.probe(video_path)
    info_dict = next(c for c in info['streams'] if c['codec_type'] == 'video')
    return int(info_dict['nb_frames']) + 1 # 经过验证，发现官方给的帧数总是比检测到的数目多1

targets_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets/'
video_dir = '/data2/hzp/Aff-Wild2/videos/'

# 所有需要分成两个视频的数据（包括训练、验证和测试集）
special_videos = {}
special_videos['10-60-1280x720'] = ['10-60-1280x720', '10-60-1280x720_right']
special_videos['video59'] = ['video59', 'video59_right']
special_videos['video2'] = ['video2', 'video2_left']
special_videos['30-30-1920x1080'] = ['30-30-1920x1080_left', '30-30-1920x1080_right']
special_videos['46-30-484x360'] = ['46-30-484x360_left', '46-30-484x360_right']
special_videos['52-30-1280x720'] = ['52-30-1280x720_left', '52-30-1280x720_right']
special_videos['135-24-1920x1080'] = ['135-24-1920x1080_left', '135-24-1920x1080_right']
special_videos['video55'] = ['video55_left', 'video55_right']
special_videos['video74'] = ['video74_left', 'video74_right']
special_videos['130-25-1280x720'] = ['130-25-1280x720_left', '130-25-1280x720_right']
special_videos['49-30-1280x720'] = ['49-30-1280x720_left', '49-30-1280x720_right']
special_videos['6-30-1920x1080'] = ['6-30-1920x1080_left', '6-30-1920x1080_right']
special_videos['video10_1'] = ['video10_1_left', 'video10_1_right']
special_videos['video29'] = ['video29_left', 'video29_right']
special_videos['video49'] = ['video49_left', 'video49_right']
special_videos['video5'] = ['video5_left', 'video5_right']

reverse_dict = {}
for key in special_videos.keys():
    for value in special_videos[key]:
        reverse_dict[value] = key

set_list = ['train', 'val']
for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))
    all_targets_path = os.path.join(targets_dir, '{}_original_all_targets.h5'.format(set_name))
    all_targets_h5f = h5py.File(all_targets_path, 'r')
    for video in tqdm(all_targets_h5f.keys()):
        record_len = all_targets_h5f[video]['length'][()]
        corr_video = reverse_dict[video] if video in reverse_dict.keys() else video
        corr_video_path = os.path.join(video_dir, corr_video + '.mp4')
        if not os.path.exists(corr_video_path):
            corr_video_path = os.path.join(video_dir, corr_video + '.avi')
        assert os.path.exists(corr_video_path)
        detect_len = get_nb_frames(corr_video_path)
        # detect_len = get_nb_frames_opencv(corr_video_path)

        # assert record_len == detect_len
        if record_len != detect_len:
            # print(record_len, detect_len)
            print(video)