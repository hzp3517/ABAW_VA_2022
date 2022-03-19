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


origin_set_name = 'Pseudo_Test_Set' #注意确认名称！！！
set_name = 'pseudo_test'
# origin_set_name = 'Test_Set' #注意确认名称！！！
# set_name = 'test'



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

dataset_root = '/data2/hzp/Aff-Wild2/'
save_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets'
mkdir(save_dir)

valid_targets_path = os.path.join(save_dir, '{}_valid_targets.h5'.format(set_name))
valid_targets_h5f = h5py.File(valid_targets_path, 'w')

origin_label_dir = os.path.join(dataset_root, 'Third_ABAW_Annotations/VA_Estimation_Challenge', origin_set_name)
txt_file_list = glob.glob(os.path.join(origin_label_dir, '*.txt'))
txt_file_list = sorted(txt_file_list)

for txt_file in tqdm(txt_file_list):
    video_id = get_basename(txt_file)
    video_group = valid_targets_h5f.create_group(video_id)

    # valence_list = []
    # arousal_list = []
    # valid_list = []

    with open(txt_file, 'r') as f:
        context = f.readlines()
    context = context[1:] # 去除首行
    label_list = [i.strip() for i in context]

    # for lb in label_list:
    #     valence_value = float(lb.split(',')[0])
    #     arousal_value = float(lb.split(',')[1])
    #     valence_list.append(valence_value)
    #     arousal_list.append(arousal_value)

    #     # check 发现没有这样的数据。
    #     if (valence_value==-5 and arousal_value!=-5) or (valence_value!=-5 and arousal_value==-5):
    #         print('notice!!!')

    #     if valence_value==-5:
    #         valid_list.append(0)
    #     else:
    #         valid_list.append(1)

    # video_group['valence'] = np.array(valence_list, dtype=np.float32)
    # video_group['arousal'] = np.array(arousal_list, dtype=np.float32)
    # video_group['valid'] = np.array(valid_list, dtype=np.int32)

    video_group['special'] = np.zeros(len(label_list)).astype(np.int)
    video_group['length'] = len(label_list)

    # assert len(label_list) == len(video_group['valence']) == len(video_group['arousal']) == len(video_group['valid'])