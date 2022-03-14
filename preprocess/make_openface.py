import os
import os.path as osp
import json
import numpy as np
import pandas as pd
import h5py 
from tqdm import tqdm
import csv
import glob

# csv_path = '/data8/hzp/tmp/00072.csv'

# with open(csv_path, 'r') as f:
#     reader = csv.reader(f)
#     # next(reader) #跳过标题行

#     title = next(reader)
#     print(title[676]) # AU01_r
#     print(title[692]) # AU45_r

#     print(title[693]) # AU01_c
#     print(title[710]) # AU45_c

#     print(title[293]) # pose_Rx --head pose
#     print(title[295]) # pose_Rz --head pose


#     print(title[2], title[9]) # gaze_0_x, gaze_angle_y
#     print(title[10], title[65]) # eye_lmk_x_0, eye_lmk_x_55
#     print(title[66], title[121]) # eye_lmk_y_0, eye_lmk_y_55

#     print(title[296]) # x_0
#     print(title[312]) # x_16
#     print(title[381], title[390]) # y_17 y_26
#     print(title[397]) # y_33

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


def normalize(c_array, min_c, max_c):
    # if max_c < min_c:##
    #     print(max_c, min_c)##
    assert max_c >= min_c
    if max_c == min_c:
        return np.zeros(len(c_array))
    else:
        return (c_array - min_c) / (max_c - min_c)


def get_openface_ft(num_frames, video_id, csv_root):
    csv_dir = osp.join(csv_root, str(video_id))
    frame_id_list = [str(i).zfill(5) for i in range(1, num_frames + 1)]

    FAU_ft_list = []
    headpose_ft_list = []
    eyegaze_ft_list = []

    for frame_id in frame_id_list:
        csv_path = os.path.join(csv_dir, '{}.csv'.format(frame_id))

        if os.path.exists(csv_path):
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                next(reader) #跳过标题行
                line = next(reader)

                hp_ft = np.array(line[293: 296]).astype(np.float32)
                FAU_ft = np.array(line[676: 711]).astype(np.float32)
                gaze_ft = np.array(line[2: 10]).astype(np.float32)
                el_x = np.array(line[10: 66]).astype(np.float32)
                el_y = np.array(line[66: 122]).astype(np.float32)
                x_0 = np.float32(line[296]) # x_min
                x_16 = np.float32(line[312]) # x_max
                y_17_26 = np.array(line[381: 391]).astype(np.float32) # for y_min
                y_33 = np.float32(line[397]) # y_max
                y_min = np.min(y_17_26)
                el_x = normalize(el_x, x_0, x_16)
                el_y = normalize(el_y, y_min, y_33)
                eg_ft = [gaze_ft, el_x, el_y]
                eg_ft = np.concatenate(eg_ft)

                FAU_ft_list.append(FAU_ft) # (35,)
                headpose_ft_list.append(hp_ft) # (3,)
                eyegaze_ft_list.append(eg_ft) # (120,)
        else:
            FAU_ft_list.append(np.zeros(35))
            headpose_ft_list.append(np.zeros(3))
            eyegaze_ft_list.append(np.zeros(120))

    FAU_fts = np.stack(FAU_ft_list)
    hp_fts = np.stack(headpose_ft_list)
    eg_fts = np.stack(eyegaze_ft_list)
    assert len(FAU_fts) == num_frames

    return FAU_fts, hp_fts, eg_fts


def make_openface(target_dir, csv_root, save_dir): # FAU, head_pose, eye_gaze
    set_list = ['train', 'val']
    special_targets_path = os.path.join(target_dir, 'special_videos.h5')
    special_h5f = h5py.File(special_targets_path, 'r')
    for set_name in set_list:
        print('--------------process {}--------------'.format(set_name))
        original_targets_path = os.path.join(target_dir, '{}_original_all_targets.h5'.format(set_name))
        valid_targets_path = os.path.join(target_dir, '{}_valid_targets.h5'.format(set_name))
        original_h5f = h5py.File(original_targets_path, 'r')
        valid_h5f = h5py.File(valid_targets_path, 'r')

        FAU_h5f = h5py.File(osp.join(save_dir, '{}_FAU.h5'.format(set_name)), 'w')
        hp_h5f = h5py.File(osp.join(save_dir, '{}_head_pose.h5'.format(set_name)), 'w')
        eg_h5f = h5py.File(osp.join(save_dir, '{}_eye_gaze.h5'.format(set_name)), 'w')

        valid_video_list = list(valid_h5f.keys())
        for new_video_id in tqdm(valid_video_list):
            FAU_group = FAU_h5f.create_group(new_video_id)
            hp_group = hp_h5f.create_group(new_video_id)
            eg_group = eg_h5f.create_group(new_video_id)
            
            if valid_h5f[new_video_id]['special'][()] == 0: # 没有被切
                num_frames = valid_h5f[new_video_id]['length'][()]
                FAU_ft, hp_ft, eg_ft = get_openface_ft(num_frames, new_video_id, csv_root) # 对齐到标签的长度
                FAU_group['fts'] = FAU_ft
                hp_group['fts'] = hp_ft
                eg_group['fts'] = eg_ft
            else: # 后切出来的片段
                original_video = '_'.join(new_video_id.split('_')[:-1])
                num_frames = original_h5f[original_video]['length'][()]
                FAU_ft, hp_ft, eg_ft = get_openface_ft(num_frames, original_video, csv_root) # 对齐到标签的长度
                seg_start = special_h5f[original_video][new_video_id]['start'][()]
                seg_end = special_h5f[original_video][new_video_id]['end'][()]
                FAU_group['fts'] = FAU_ft[seg_start: seg_end + 1]
                hp_group['fts'] = hp_ft[seg_start: seg_end + 1]
                eg_group['fts'] = eg_ft[seg_start: seg_end + 1]
                assert len(FAU_group['fts']) == valid_h5f[new_video_id]['length'][()]


if __name__ == '__main__':
    # face_root = '/data9/datasets/Aff-Wild2/cropped_aligned/'
    target_dir = '/data9/hzp/ABAW_VA_2022/processed_data/targets/'
    csv_root = '/data9/hzp/ABAW_VA_2022/processed_data/openface_save/'
    save_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features/'
    mkdir(save_dir)
    print('making openface')
    make_openface(target_dir, csv_root, save_dir)