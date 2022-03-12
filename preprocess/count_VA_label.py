import os
import numpy as np
import h5py
from tqdm import tqdm

target_dir = '/data9/hzp/ABAW_VA_2022/processed_data/targets/'
set_list = ['train', 'val']

valence_cnt = {}
arousal_cnt = {}
range_str_lst = []

for i in range(19):
    range_str = '[{}~{})'.format(round(i/10.0-1, 4), round((i+1)/10.0-1, 4))
    range_str_lst.append(range_str)
    valence_cnt[range_str] = 0
    arousal_cnt[range_str] = 0
    
range_str = '[{}~{}]'.format(round(19/10.0-1, 4), round(20/10.0-1, 4))
range_str_lst.append(range_str)
valence_cnt[range_str] = 0
arousal_cnt[range_str] = 0

def transform(x):
    if x == 1:
        return range_str_lst[19]
    else:
        x += 1
        x *= 10
        x = int(x)
        return range_str_lst[x]

for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))
    valid_targets_path = os.path.join(target_dir, '{}_valid_targets.h5'.format(set_name))
    valid_targets_h5f = h5py.File(valid_targets_path, 'r')
    for video in tqdm(list(valid_targets_h5f.keys())):
        valence_list = valid_targets_h5f[video]['valence']
        arousal_list = valid_targets_h5f[video]['arousal']
        for i in range(len(valence_list)):
            valence_idx = transform(valence_list[i])
            arousal_idx = transform(arousal_list[i])
            valence_cnt[valence_idx] += 1
            arousal_cnt[arousal_idx] += 1
            
    print('valence:')
    for k in valence_cnt.keys():
        print('{}:\t{}'.format(k, valence_cnt[k]))
        
    print('arousal:')
    for k in arousal_cnt.keys():
        print('{}:\t{}'.format(k, arousal_cnt[k]))
    
# if __name__ == '__main__':
#     print(transform(-0.23))
    
    
    
    
    