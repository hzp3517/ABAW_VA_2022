'''
读入图像后先转换为灰度图，然后resize成64x64（适配于densenet）
需要补全缺失的帧（这里是前后找最近的帧补（先找前面，再找后面））。
以video-level存成h5文件
{
    [new_video_id]: 
    {
        'images': [img (64*64), img, ...],
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


origin_set_name = 'Pseudo_Test_Set' #注意确认名称！！！
set_name = 'pseudo_test'
# origin_set_name = 'Test_Set' #注意确认名称！！！
# set_name = 'test'



dataset_root = '/data2/hzp/Aff-Wild2/'
features_dir = '/data2/hzp/ABAW_VA_2022/processed_data/features'
targets_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets'
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

def convert_img(img_path, img_size=64):
    '''
    return:
    - success
    - img
    '''
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (img_size, img_size))
    # img = img / 255.0
    assert img.shape == (img_size, img_size)
    return img

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

    # for frame_id in video_frame_ids:
    for idx in range(len(video_frame_ids)):
        frame_id = video_frame_ids[idx]
        img_path = os.path.join(img_path_dir, frame_id + '.jpg')
        if os.path.exists(img_path):
            img_data = convert_img(img_path)
            pad.append(0)
            video_data.append(img_data)
        else:
            # 图片不存在，寻找附近帧
            delta = 1
            while 1:
                if idx - delta > 0:
                    former_img_path = os.path.join(img_path_dir, video_frame_ids[idx-delta] + '.jpg')
                else:
                    former_img_path = None
                    
                if former_img_path and os.path.exists(former_img_path):
                    img_data = convert_img(former_img_path)
                    pad.append(1)
                    video_data.append(img_data)
                    break

                if idx + delta < len(video_frame_ids):
                    later_img_path = os.path.join(img_path_dir, video_frame_ids[idx+delta] + '.jpg')
                else:
                    later_img_path = None

                if later_img_path and os.path.exists(later_img_path):
                    img_data = convert_img(later_img_path)
                    pad.append(1)
                    video_data.append(img_data)
                    break
                
                delta += 1
        
    assert len(video_data) == valid_len_video
    assert len(pad) == valid_len_video
    return video_data, pad

mkdir(features_dir)

valid_targets_path = os.path.join(targets_dir, '{}_valid_targets.h5'.format(set_name))
valid_targets_h5f = h5py.File(valid_targets_path, 'r')
gray_img_path = os.path.join(features_dir, '{}_gray_img_data.h5'.format(set_name))
gray_img_h5f = h5py.File(gray_img_path, 'w')

for video in tqdm(list(valid_targets_h5f.keys())):
    video_group = gray_img_h5f.create_group(video)
    video_img_dir = os.path.join(cropped_aligned_root, video)
    assert os.path.exists(video_img_dir)
    video_len = valid_targets_h5f[video]['length'][()]
    video_frame_ids = [str(i).zfill(5) for i in range(1, video_len + 1)]
    video_data, pad = get_video_data(video, video_frame_ids)
    video_group['images'] = np.array(video_data).astype(np.float32)
    assert len(video_group['images']) == video_len
    video_group['pad'] = np.array(pad)
