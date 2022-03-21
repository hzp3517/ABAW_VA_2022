import h5py
import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code')

train_path = '/data2/hzp/ABAW_VA_2022/processed_data/targets/train_valid_targets.h5'
val_path = '/data2/hzp/ABAW_VA_2022/processed_data/targets/val_valid_targets.h5'

train_h5f = h5py.File(val_path, 'r')
for video in tqdm(train_h5f.keys()):
    if np.any(train_h5f[video]['valence'][()]) == False:
        print('valence: {}'.format(video))
    if np.any(train_h5f[video]['arousal'][()]) == False:
        print('arousal: {}'.format(video))
