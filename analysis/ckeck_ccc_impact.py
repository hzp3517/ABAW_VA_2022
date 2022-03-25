import h5py
import csv
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import json
import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code')
from utils.metrics import evaluate_regression, remove_padding, scratch_data, smooth_func, smooth_predictions

json_path = '/data2/hzp/ABAW_VA_2022/code/test_results/3-16/transformer_both_affectnet-vggish-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc/val_pred_arousal_nosmooth.json'

with open(json_path, 'r') as f:
    json_dict = json.load(f)

video_ids = list(json_dict.keys())

total_preds = []
total_labels = []

modified_total_preds = []
modified_total_labels = []

for video in tqdm(video_ids):
    total_preds.append(json_dict[video]['pred'])
    total_labels.append(json_dict[video]['label'])

    modified_total_preds.append(json_dict[video]['pred'][2:])
    modified_total_labels.append(json_dict[video]['label'][:-2])

# print(len(total_preds))
total_preds = scratch_data(total_preds)
total_labels = scratch_data(total_labels)
modified_total_preds = scratch_data(modified_total_preds)
modified_total_labels = scratch_data(modified_total_labels)

_, _, _, origin_ccc = evaluate_regression(total_labels, total_preds)
_, _, _, modified_ccc = evaluate_regression(modified_total_labels, modified_total_preds)

print(origin_ccc, modified_ccc)

