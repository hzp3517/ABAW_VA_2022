'''
读入图像后resize成224×224（还是3通道）（适配于resnet-50）
需要补全缺失的帧（默认从邻近的前面的帧补，如果缺的是第一帧，就从后面的最靠前的有效帧补）。
以video-level存成h5文件
{
    [new_video_id]: 
    {
        'images': [img (224*224*3), img, ...],
        'pad': [0, 0, 1, ...] # 0表示原始图像有效，1表示该帧为从旁边帧填充
    }
}
'''
import os
import h5py
import numpy as np
import glob
from tqdm import tqdm
import cv2

dataset_root = '/data9/datasets/Aff-Wild2/'
features_dir = '/data9/hzp/ABAW_VA_2022/processed_data/features'
targets_dir = '/data9/hzp/ABAW_VA_2022/processed_data/targets'
cropped_aligned_root = os.path.join(dataset_root, 'cropped_aligned')

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

def convert_img(img_path, img_size=224):
    '''
    return:
    - success
    - img
    '''
    if os.path.exists(img_path) == False:
        return False, None
    img = cv2.imread(img_path)
    if img is None:
        return False, None
    img = cv2.resize(img, (img_size, img_size))
    # img = img / 255.0
    assert img.shape == (img_size, img_size, 3)
    return True, img

def get_video_data(video_id, video_frame_ids):
    '''
    video_frame_ids: [00001, ...]
    
    return:
    - video_data
    - pad: [0, 0, 1, ...] # 0表示使用原始图像，1表示使用邻近图像填充
    '''
    valid_len_video = len(video_frame_ids)
    pad = []
    
    video_data = []
    img_path_dir = os.path.join(cropped_aligned_root, video_id)
    
    first_img_path = os.path.join(img_path_dir, video_frame_ids[0] + '.jpg')
    success, first_img_data = convert_img(first_img_path)
    video_frame_ids = video_frame_ids[1:]

    # 如果第一帧不存在，就用后面存在的最靠前的帧补
    if success == False:
        pad.append(1)
        for img in video_frame_ids:
            img_path = os.path.join(img_path_dir, img + '.jpg')
            success, img_data = convert_img(img_path)
            if success:
                video_data.append(img_data)
                break
    else:
        pad.append(0)
        video_data.append(first_img_data)
    assert len(video_data) == 1
    
    # 后续的帧如果缺失，就用其前面一个有效帧（标签不为-1）补充
    for frame_id in video_frame_ids:
        img_path = os.path.join(img_path_dir, frame_id + '.jpg')
        success, img_data = convert_img(img_path)
        if success:
            pad.append(0)
            video_data.append(img_data)
        else:
            pad.append(1)
            video_data.append(video_data[-1])
        
    assert len(video_data) == valid_len_video
    assert len(pad) == valid_len_video
    return video_data, pad

mkdir(features_dir)
origin_set_list = ['Train_Set', 'Validation_Set']
set_list = ['train', 'val']

special_file = os.path.join(targets_dir, 'special_videos.h5')
special_h5f = h5py.File(special_file, 'r')

for origin_set_name, set_name in zip(origin_set_list, set_list):
    print('--------------process {}--------------'.format(set_name))

    valid_targets_path = os.path.join(targets_dir, '{}_valid_targets.h5'.format(set_name))
    valid_targets_h5f = h5py.File(valid_targets_path, 'r')
    original_img_path = os.path.join(features_dir, '{}_original_img_data.h5'.format(set_name))
    original_img_h5f = h5py.File(original_img_path, 'w')

    for video in tqdm(list(valid_targets_h5f.keys())):
    # for video in tqdm(list(valid_targets_h5f.keys())[:2]):##

        if valid_targets_h5f[video]['special'][()] == 0:
            video_group = original_img_h5f.create_group(video)
            video_img_dir = os.path.join(cropped_aligned_root, video)
            assert os.path.exists(video_img_dir)
            video_len = valid_targets_h5f[video]['length'][()]
            video_frame_ids = [str(i).zfill(5) for i in range(1, video_len + 1)]
            video_data, pad = get_video_data(video, video_frame_ids)
            video_group['images'] = np.array(video_data).astype(np.float32)
            assert len(video_group['images']) == video_len
            video_group['pad'] = np.array(pad)
    
        else: # 后切出来的片段
            original_video = '_'.join(video.split('_')[:-1])
            video_img_dir = os.path.join(cropped_aligned_root, original_video)
            assert os.path.exists(video_img_dir)

            video_group = original_img_h5f.create_group(video)
            seg_start = special_h5f[original_video][video]['start'][()] + 1
            seg_end = special_h5f[original_video][video]['end'][()] + 1
            seg_frame_ids = [str(i).zfill(5) for i in range(seg_start, seg_end + 1)]
            seg_data, pad = get_video_data(original_video, seg_frame_ids)
            video_group['images'] = np.array(seg_data).astype(np.float32)
            assert len(video_group['images']) == special_h5f[original_video][video]['length'][()]
            video_group['pad'] = np.array(pad)





# if __name__ == '__main__':
#     img_path = '/data9/datasets/Aff-Wild2/cropped_aligned/1-30-1280x720/00001.jpg'
#     success, img = convert_img(img_path)
#     print(img)
#     print(img.shape)

