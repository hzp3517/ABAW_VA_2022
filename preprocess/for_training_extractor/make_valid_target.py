'''
[set]_targets.h5
{
    [video_id]:
    {
        'frame': [1, 2, 4, 6, ...] # 分别对应于00001, 00002, ...
        'label': [[0.2, 0.3], [], [], [], ...] (valence, arousal)
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

origin_target_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets/'
save_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets/for_training_extractor'
image_root = '/data2/hzp/Aff-Wild2/cropped_aligned/'
mkdir(save_dir)

set_list = ['train', 'val']

for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))
    ori_target_path = os.path.join(origin_target_dir, '{}_original_all_targets.h5'.format(set_name))
    ori_target_h5f = h5py.File(ori_target_path, 'r')
    save_path = os.path.join(save_dir, '{}_targets.h5'.format(set_name))
    save_h5f = h5py.File(save_path, 'w')

    for video in tqdm(list(ori_target_h5f.keys())):
        valence_list = ori_target_h5f[video]['valence'][()]
        arousal_list = ori_target_h5f[video]['arousal'][()]
        video_group = save_h5f.create_group(video)

        frame_list = []
        label_list = []

        for frame_idx in range(len(valence_list)):

            if valence_list[frame_idx] > -5:
                frame_no = str(frame_idx + 1).zfill(5)
                frame_path = os.path.join(image_root, video, frame_no + '.jpg')

                if os.path.exists(frame_path):
                    valence = valence_list[frame_idx]
                    arousal = arousal_list[frame_idx]

                    frame_list.append(frame_idx + 1)
                    label_list.append([valence, arousal])

        assert len(frame_list) == len(label_list)

        video_group['frame'] = np.array(frame_list)
        video_group['label'] = np.array(label_list, dtype=np.float32)
