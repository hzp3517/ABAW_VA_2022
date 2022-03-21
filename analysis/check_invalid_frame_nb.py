import h5py
import os
from tqdm import tqdm
import numpy as np

ori_targets_file = '/data2/hzp/ABAW_VA_2022/processed_data/targets/val_original_all_targets.h5'

invalid_cnt = 0

val_h5f = h5py.File(ori_targets_file, 'r')
for video in tqdm(val_h5f.keys()):
    valence = val_h5f[video]['valence'][()]
    for value in valence:
        if value == -5:
            invalid_cnt += 1


print(invalid_cnt)
