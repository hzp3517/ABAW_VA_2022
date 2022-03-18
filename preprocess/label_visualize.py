import h5py
import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass


targets_dir = '/data2/hzp/ABAW_VA_2022/processed_data/targets/'
label_csv_root = os.path.join(targets_dir, 'label_csv')
set_list = ['train', 'val']

for set_name in set_list:
    print('--------------process {}--------------'.format(set_name))
    set_csv_dir = os.path.join(label_csv_root, set_name)
    mkdir(set_csv_dir)
    h5f = h5py.File(os.path.join(targets_dir, '{}_valid_targets.h5'.format(set_name)))
    for video in tqdm(list(h5f.keys())):
        valence = h5f[video]['valence'][()]
        arousal = h5f[video]['arousal'][()]
        frame_ids = np.array([i for i in range(1, len(valence) + 1)])
        df = pd.DataFrame({'frame_id': frame_ids, 'valence': valence, 'arousal': arousal})
        csv_path = os.path.join(set_csv_dir, '{}.csv'.format(video))
        df.to_csv(csv_path, index=False, sep=',')

