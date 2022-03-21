import h5py
import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code')
from utils.metrics import evaluate_regression

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

pred_csv_root = '/data2/hzp/ABAW_VA_2022/code/pred_csv'
mkdir(pred_csv_root)

test_results_dir_list = [
    '/data2/hzp/ABAW_VA_2022/code/test_results/3-16/transformer_lstm/transformer_lstm_both_affectnet-vggish-wav2vec_res-n_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc_run3'
]

for test_results_dir in tqdm(test_results_dir_list):
    if os.path.exists(os.path.join(test_results_dir.strip(), 'val_pred_arousal_nosmooth.json')):
        target = 'arousal'
    else:
        target = 'valence'
    json_list = ['val_pred_{}_nosmooth.json'.format(target), 'val_pred_{}_smooth.json'.format(target)]
    for json_file in json_list:
        video_ccc_dict = {}

        csv_dir = os.path.join(pred_csv_root, json_file.split('.')[0])
        mkdir(csv_dir)
        json_path = os.path.join(test_results_dir, json_file)
        with open(json_path, 'r') as f:
            json_dict = json.load(f)
        for video in json_dict.keys():
            video_pred = json_dict[video]['pred']
            video_label = json_dict[video]['label'] # 其中，video58的标注全是0

            # print('lable: ')
            # print(np.array(video_label))
            # print('pred: ')
            # print(np.array(video_pred))

            if np.any(np.array(video_label)) == False:
                print(video)

            if video != 'video'
            mse, rmse, pcc, ccc = evaluate_regression(np.array(video_label), np.array(video_pred))
            video_ccc_dict[video] = ccc
            df = pd.DataFrame({'pred': video_pred, target: video_label})
            csv_path = os.path.join(csv_dir, '{}.csv'.format(video))
            df.to_csv(csv_path, index=False, sep=',')

        ordered_ccc_dict = sorted(video_ccc_dict.items(),key=lambda x:x[1], reverse=True)

        txt_path = os.path.join(csv_dir, 'ccc_results.txt')
        with open(txt_path, 'w') as f:
            for k, v in ordered_ccc_dict:
                f.write('{}:\t{}\n'.format(k, v))