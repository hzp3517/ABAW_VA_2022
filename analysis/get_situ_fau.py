import os
import csv
import numpy as np
import pandas as pd

situ_dir = '/data2/hzp/ABAW_VA_2022/processed_data/situ_fau/situ_fau/'
set_list = ['train', 'val']

for set_name in set_list:
    csv_path = os.path.join(situ_dir, set_name+'.csv')
    npy_path = os.path.join(situ_dir, set_name+'.npy')

    csv_data = pd.read_csv(csv_path)
    npy_data = np.load(npy_path)

    # embedd_idx = csv_data['embedd_idx']

    # # print(embedd_idx[:10000])
    # # print(npy_data.shape)

    # break

