'''
把所有视频的所有标注都存到一个h5文件中
{
    [video_id]: {
        'special': 0, # 1表示这一段是切出来之后的片段，0表示这个视频就是原本对应的id
        'length': xx
    }
}
'''
import os
import h5py
import numpy as np
import glob
from tqdm import tqdm
import ffmpeg

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

def get_nb_frames(video_path):
    info = ffmpeg.probe(video_path)
    info_dict = next(c for c in info['streams'] if c['codec_type'] == 'video')
    return int(info_dict['nb_frames']) + 1 # 经过验证，发现官方给的帧数总是比检测到的数目多1

test_label_file = '/data2/hzp/Aff-Wild2/Test_Set_Release_and_Submission_Instructions/Valence_Arousal_Estimation_Challenge_test_set_release.txt'
set_name = 'test'

# dataset_root = '/data2/hzp/Aff-Wild2/'
video_dir = '/data2/hzp/Aff-Wild2/videos/'
save_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets'
mkdir(save_dir)

with open(test_label_file, 'r') as f:
    context = f.readlines()
test_video_list = [i.strip() for i in context]

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
    
valid_targets_path = os.path.join(save_dir, '{}_valid_targets.h5'.format(set_name))
valid_targets_h5f = h5py.File(valid_targets_path, 'w')

for video in tqdm(test_video_list):
    video_group = valid_targets_h5f.create_group(video)
    if video in reverse_dict.keys():
        corr_video = reverse_dict[video]
        video_path = os.path.join(video_dir, corr_video + '.mp4')
        if not os.path.exists(video_path):
            video_path = os.path.join(video_dir, corr_video + '.avi')
        assert os.path.exists(video_path)
        nb_frames = get_nb_frames(video_path)
        video_group['special'] = np.zeros(nb_frames).astype(np.int32)
        video_group['length'] = nb_frames
    else:
        video_path = os.path.join(video_dir, video + '.mp4')
        if not os.path.exists(video_path):
            video_path = os.path.join(video_dir, video + '.avi')
        assert os.path.exists(video_path)
        nb_frames = get_nb_frames(video_path)
        video_group['special'] = np.zeros(nb_frames).astype(np.int32)
        video_group['length'] = nb_frames