'''
得到wav2vec特征

wav2vec特征是0.02s一帧，标签频率是0.0333s一帧
'''

import numpy as np
import h5py
import os
from tqdm import tqdm
import math

import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code/preprocess')
from tools.wav2vec import Wav2VecExtractor


set_name = 'pseudo_test' # 注意改过来！！！
# set_name = 'test'



def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def get_wav2vec_ft(extractor, audio_root, video_id, num_frames, gpu_id, frame_rate=0.02):
    '''
    num_frames: 30Hz
    '''
    wav2vec = extractor # Wav2VecExtractor(gpu=gpu_id) # 不做降采样，0.02s一帧特征
    wav_path = os.path.join(audio_root, video_id + '.wav')
    origin_ft = wav2vec(wav_path) # (len, 130)
    origin_ft = np.array(origin_ft)
    
    start_time_list = [i * (1.0/30) for i in range(num_frames)]
    end_time_list = [((1.0/30) + i * (1.0/30)) for i in range(num_frames)]
    
    # 把每个时刻的值精度缩小一下，防止因为float本身有些小数最后一位不够精确导致多算一条frame
    start_time_list = [round(i, 4) for i in start_time_list]
    end_time_list = [round(i, 4) for i in end_time_list]
    
    # 确保 origin_ft pad到足够长
    finish_idx = math.ceil(end_time_list[-1] / frame_rate)
    if origin_ft.shape[0] < finish_idx:
        pad_ft = []
        pad_num = finish_idx + 1 - origin_ft.shape[0]
        for i in range(pad_num):
            pad_ft.append(origin_ft[-1])
        pad_ft = np.stack(pad_ft)
        origin_ft = np.concatenate((origin_ft, pad_ft), axis=0)
    
    video_ft = []
    for start, end in zip(start_time_list, end_time_list):
        start_idx = math.floor(round(start / frame_rate, 4)) # 同样也需要在这里缩小一下精度
        end_idx = math.ceil(round(end / frame_rate, 4))
        frame_ft = origin_ft[start_idx: end_idx]
        frame_ft = np.mean(frame_ft, axis=0)
        video_ft.append(frame_ft)
    video_ft = np.stack(video_ft, axis=0).astype(np.float32)
    assert len(video_ft) == num_frames
    
    return video_ft


def make_wav2vec(target_dir, audio_root, save_dir, gpu_id):
    extractor = Wav2VecExtractor(gpu=gpu_id)
    # special_targets_path = os.path.join(target_dir, 'special_videos.h5')
    # special_h5f = h5py.File(special_targets_path, 'r')
    

    # original_targets_path = os.path.join(target_dir, '{}_valid_targets.h5'.format(set_name))
    valid_targets_path = os.path.join(target_dir, '{}_valid_targets.h5'.format(set_name))
    ft_path = os.path.join(save_dir, '{}_wav2vec.h5'.format(set_name))
    # original_h5f = h5py.File(original_targets_path, 'r')
    valid_h5f = h5py.File(valid_targets_path, 'r')
    ft_h5f = h5py.File(ft_path, 'w')

    valid_video_list = list(valid_h5f.keys())
    for new_video_id in tqdm(valid_video_list):
        video_group = ft_h5f.create_group(new_video_id)
        
        # if valid_h5f[new_video_id]['special'][()] == 0: # 没有被切
        num_frames = valid_h5f[new_video_id]['length'][()]
        video_ft = get_wav2vec_ft(extractor, audio_root, new_video_id, num_frames, gpu_id)
        video_group['fts'] = video_ft

        # else: # 后切出来的片段
        #     original_video = '_'.join(new_video_id.split('_')[:-1])
        #     num_frames = original_h5f[original_video]['length'][()]
        #     video_ft = get_wav2vec_ft(extractor, audio_root, original_video, num_frames, gpu_id)
        #     seg_start = special_h5f[original_video][new_video_id]['start'][()]
        #     seg_end = special_h5f[original_video][new_video_id]['end'][()]
        #     video_group['fts'] = video_ft[seg_start: seg_end + 1]
        #     assert len(video_group['fts']) == valid_h5f[new_video_id]['length'][()]
                

if __name__ == '__main__':
    audio_root = '/data2/hzp/ABAW_VA_2022/processed_data/audios/'
    target_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets/'
    save_dir = '/data2/hzp/ABAW_VA_2022/processed_data/features/'
    mkdir(save_dir)
    print('making Wav2Vec')
    make_wav2vec(target_dir, audio_root, save_dir, gpu_id=0)