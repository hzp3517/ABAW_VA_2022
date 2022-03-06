'''
复制所有语音文件到一个新的目录，主要是为了把含有_left或_right的语音复制好
'''
import os
import h5py
from tqdm import tqdm


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

origin_audio_dir = '/data9/hzp/ABAW_VA_2022/processed_data/original_audios/'
new_audio_dir = '/data9/hzp/ABAW_VA_2022/processed_data/audios/'
set_list = ['train', 'val']

mkdir(new_audio_dir)
origin_target_list = []

for set_name in set_list:
    original_targets_path = '/data9/hzp/ABAW_VA_2022/processed_data/targets/{}_original_all_targets.h5'.format(set_name)
    original_h5f = h5py.File(original_targets_path, 'r')
    origin_target_list += list(original_h5f.keys())

original_audio_path_list = [os.path.join(origin_audio_dir, i) for i in os.listdir(origin_audio_dir)]
for origin_path in tqdm(original_audio_path_list):
    audio_name = get_basename(origin_path)
    if audio_name in special_videos.keys():
        save_path_1 = os.path.join(new_audio_dir, special_videos[audio_name][0] + '.wav')
        save_path_2 = os.path.join(new_audio_dir, special_videos[audio_name][1] + '.wav')
        cmd_str_1 = 'cp {} {}'.format(origin_path, save_path_1)
        cmd_str_2 = 'cp {} {}'.format(origin_path, save_path_2)
        os.system(cmd_str_1)
        os.system(cmd_str_2)
    else:
        save_path = os.path.join(new_audio_dir, audio_name + '.wav')
        cmd_str = 'cp {} {}'.format(origin_path, save_path)
        os.system(cmd_str)

new_audio_path_list = [os.path.join(new_audio_dir, i) for i in os.listdir(new_audio_dir)]
new_audio_list = [get_basename(i) for i in new_audio_path_list]
for audio in tqdm(origin_target_list):
    assert audio in new_audio_list
    

