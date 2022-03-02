'''
将3个含有无效标注的视频分成新的几段有效标注段
{
    [original_video_id]: {
        [new_video_id]: {
            'start': 5, # 注意：id是从0开始的！
            'end': 9,
            'length': 5
        }
    }
}
'''
import os
import h5py
import numpy as np
import glob
from tqdm import tqdm

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

target_dir = '/data9/hzp/ABAW_VA_2022/processed_data/targets'
# set_list = ['train', 'val']

trn_problem_videos = ['10-60-1280x720_right', '86-24-1920x1080']
val_problem_video = 'video59_right'

speical_videos_path = os.path.join(target_dir, 'special_videos.h5')
special_videos_h5f = h5py.File(speical_videos_path, 'w')

# trn
all_targets_path = os.path.join(target_dir, '{}_original_all_targets.h5'.format('train'))
all_targets_h5f = h5py.File(all_targets_path, 'r')
for video in trn_problem_videos:
    print('-------------video {}---------------'.format(video))
    video_group = special_videos_h5f.create_group(video)
    valid = all_targets_h5f[video]['valid'][()]
    cnt_seg = 0
    start = 0
    end = 0

    i = 0
    while i < len(valid):
        if valid[i] != 0:
            i += 1
        elif i == 0:
            while i < len(valid):
                if valid[i] != 0:
                    start = i
                    i += 1
                    break
                i += 1
        else:
            end = i - 1
            cnt_seg += 1
            new_video_group = video_group.create_group('{}_{}'.format(video, cnt_seg))
            new_video_group['start'] = start
            new_video_group['end'] = end
            new_video_group['length'] = end - start + 1

            while i < len(valid):
                if valid[i] != 0:
                    start = i
                    i += 1
                    break
                i += 1

    if valid[-1] != 0:
        end = i - 1
        cnt_seg += 1
        new_video_group = video_group.create_group('{}_{}'.format(video, cnt_seg))
        new_video_group['start'] = start
        new_video_group['end'] = end
        new_video_group['length'] = end - start + 1


# val
video = val_problem_video
video_group = special_videos_h5f.create_group(video)
print('-------------video {}---------------'.format(video))
all_targets_path = os.path.join(target_dir, '{}_original_all_targets.h5'.format('val'))
all_targets_h5f = h5py.File(all_targets_path, 'r')
valid = all_targets_h5f[video]['valid']

cnt_seg = 0
start = 0
end = 0

i = 0
while i < len(valid):
    if valid[i] != 0:
        i += 1
    elif i == 0:
        while i < len(valid):
            if valid[i] != 0:
                start = i
                i += 1
                break
            i += 1
    else:
        end = i - 1
        cnt_seg += 1
        new_video_group = video_group.create_group('{}_{}'.format(video, cnt_seg))
        new_video_group['start'] = start
        new_video_group['end'] = end
        new_video_group['length'] = end - start + 1

        while i < len(valid):
            if valid[i] != 0:
                start = i
                i += 1
                break
            i += 1

if valid[-1] != 0:
    end = i - 1
    cnt_seg += 1
    new_video_group = video_group.create_group('{}_{}'.format(video, cnt_seg))
    new_video_group['start'] = start
    new_video_group['end'] = end
    new_video_group['length'] = end - start + 1