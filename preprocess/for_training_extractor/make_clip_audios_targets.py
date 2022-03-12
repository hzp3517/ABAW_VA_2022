'''
把每个语音文件切成最长2min的片段，标签也做同样的处理
'''
import os
import glob
from tqdm import tqdm
import torch
import pandas as pd
import soundfile as sf
import numpy as np
import subprocess
import librosa
import scipy.signal as spsig
import h5py

def mkdir(path):
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except FileExistsError:
        pass
    
def get_basename(path):
    basename = os.path.basename(path)
    if os.path.isfile(path):
        basename = basename[:basename.rfind('.')]
    return basename

ori_audio_dir = '/data9/hzp/ABAW_VA_2022/processed_data/audios/'
ori_targets_dir = '/data9/hzp/ABAW_VA_2022/processed_data/targets/'
save_audio_dir = '/data9/hzp/ABAW_VA_2022/processed_data/clip_audios/'
save_targets_dir = '/data9/hzp/ABAW_VA_2022/processed_data/targets/for_training_extractor/'

mkdir(save_audio_dir)
mkdir(save_targets_dir)

set_list = ['train', 'val']
special_targets_path = os.path.join(ori_targets_dir, 'special_videos.h5')
special_h5f = h5py.File(special_targets_path, 'r')
max_seg_len = 120 # 最长语音片段2min
fps = 30


def get_audio_clip(ori_audio_file, times, clip_id):
    '''
    input:
    - ori_audio_file: original audio file
    - times: [s_time, e_time]
    - clip_id: xx # 当前片段的id，从1开始
    return:
    - clip_id: '[new_video_id]_[001]' # clip_id补到3位，保证h5文件key中的排序是按顺序的
    - save_path: 生成语音片段文件的保存路径
    '''
    new_video_id = get_basename(ori_audio_file)
    clip_id = str(clip_id).zfill(3)
    clip_id = '{}_{}'.format(new_video_id, clip_id)
    save_path = os.path.join(save_audio_dir, '{}.wav'.format(clip_id))
    # _cmd = "ffmpeg -i {} -vn -f wav -acodec pcm_s16le -ac 1 -ar 16000 {} -y > /dev/null 2>&1".format(ori_audio_file, save_path)
    _cmd = 'ffmpeg -ss {} -t {} -i {} -c:v libx264 -c:a aac -strict experimental -b:a 98k {} >/dev/null 2>&1'
    s_time = times[0]
    duration = times[1] - s_time
    os.system(_cmd.format(s_time, duration, ori_audio_file, save_path))
    return clip_id, save_path


for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))
    original_targets_path = os.path.join(ori_targets_dir, '{}_original_all_targets.h5'.format(set_name))
    valid_targets_path = os.path.join(ori_targets_dir, '{}_valid_targets.h5'.format(set_name))
    save_targets_path = os.path.join(save_targets_dir, '{}_clip_audio_targets.h5'.format(set_name))
    original_h5f = h5py.File(original_targets_path, 'r')
    valid_h5f = h5py.File(valid_targets_path, 'r')
    ft_h5f = h5py.File(save_targets_path, 'w')
    
    valid_video_list = list(valid_h5f.keys())
    for new_video_id in tqdm(valid_video_list):
        audio_path = os.path.join(ori_audio_dir, new_video_id + '.wav')
        video_group = ft_h5f.create_group(new_video_id)
        
        if valid_h5f[new_video_id]['special'][()] == 0: # 没有被切
            num_frames = valid_h5f[new_video_id]['length'][()]
            
            
            
            # video_ft = get_vggish_ft(audio_root, new_video_id, num_frames, gpu_id)
            # video_group['fts'] = video_ft
