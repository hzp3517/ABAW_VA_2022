from cgi import test
import os
import json
import numpy as np
import time
import datetime
import numpy as np
from opts.test_opts import TestOptions
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger
from utils.path import make_path
from utils.metrics import evaluate_regression, remove_padding, scratch_data, smooth_predictions
from utils.tools import calc_total_dim
# from utils.tools import get_each_dim
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from collections import OrderedDict
import torch
import sys
from data import find_dataset_using_name
import fcntl
import csv
from models.model_utils.config import OptConfig
import logging

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

logs_root = '/data2/hzp/ABAW_VA_2022/code/logs'


# ------settings-----------
test_logs="\
3-16/transformer_both_affectnet-compare-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc;\
3-16/transformer_both_affectnet-vggish-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc\
"

target_list = ['valence', 'arousal']

epoch_ids=[12, 13]

result_root = '/data2/hzp/ABAW_VA_2022/code/val_results'
mkdir(result_root)

# ---------------------------

def smooth_func(pred, label=None, best_window=None, result_path=None):
    if best_window is None:
        best_ccc, best_window = 0, 0
        
        if result_path:
            with open(result_path, 'w') as f:
                for window in range(0, 70, 5):
                    smoothed_preds = smooth_predictions(pred, window=window)
                    if label is not None:
                        mse, rmse, pcc, ccc = evaluate_regression(y_true=scratch_data(label),
                                                                y_pred=scratch_data(smoothed_preds))
                        
                        f.writelines('In smoothing \twindow {} \tmse {:.4f}, rmse {:.4f}, pcc {:.4f}, ccc {:.4f}\n'.format(
                            window, mse, rmse, pcc, ccc))
                        print('In smoothing \twindow {} \tmse {:.4f}, rmse {:.4f}, pcc {:.4f}, ccc {:.4f}'.format(
                            window, mse, rmse, pcc, ccc))
                        
                        if ccc > best_ccc:
                            best_ccc, best_window = ccc, window

                f.writelines('Smooth: best window {:d} best_ccc {:.4f}\n'.format(best_window, best_ccc))
                print('Smooth: best window {:d} best_ccc {:.4f}'.format(best_window, best_ccc))

        else:
            for window in range(0, 70, 5):
                smoothed_preds = smooth_predictions(pred, window=window)
                if label is not None:
                    mse, rmse, pcc, ccc = evaluate_regression(y_true=scratch_data(label),
                                                            y_pred=scratch_data(smoothed_preds))
                    
                    print('In smoothing \twindow {} \tmse {:.4f}, rmse {:.4f}, pcc {:.4f}, ccc {:.4f}'.format(
                        window, mse, rmse, pcc, ccc))
                    
                    if ccc > best_ccc:
                        best_ccc, best_window = ccc, window
            print('Smooth: best window {:d} best_ccc {:.4f}'.format(best_window, best_ccc))

        smoothed_preds = smooth_predictions(pred, window=best_window)
    elif best_window is not None:
        ori_mse, ori_rmse, ori_pcc, ori_ccc = evaluate_regression(y_true=scratch_data(label),
                                                            y_pred=scratch_data(pred))
        smoothed_preds = smooth_predictions(pred, window=best_window)
        mse, rmse, pcc, ccc = evaluate_regression(y_true=scratch_data(label),
                                                            y_pred=scratch_data(smoothed_preds))
        if result_path:
            with open(result_path, 'w') as f:
                f.writelines('original ccc {:.4f}\n'.format(ori_ccc))
                print('original ccc {:.4f}\n'.format(ori_ccc))
                f.writelines('Smooth: setting window {:d} ccc {:.4f}\n'.format(best_window, ccc))
                print('Smooth: setting window {:d} ccc {:.4f}'.format(best_window, ccc))
        else:
            print('original ccc {:.4f}\n'.format(ori_ccc))
            print('Smooth: setting window {:d} ccc {:.4f}'.format(best_window, ccc))


    return smoothed_preds, best_window




def eval_for_val(predictions, labels, result_path=None, is_smooth=False, best_window=None):
    '''
    predictions: [[1, 0.2, ...], [0.2, 0.2, ...], ...]
    labels: [[1, 0.2, ...], [0.2, 0.2, ...], ...]
    '''
    # calculate metrics
    if is_smooth:
        smoothed_preds, best_window = smooth_func(predictions, labels, best_window, result_path)
    else:
        best_window = None
        smoothed_preds = predictions
        mse, rmse, pcc, ccc = evaluate_regression(labels, smoothed_preds)
        with open(result_path, 'w') as f:
            f.writelines('original ccc {:.4f}\n'.format(ccc))
    smoothed_preds = scratch_data(smoothed_preds)
    # labels = scratch_data(labels)
    # mse, rmse, pcc, ccc = evaluate_regression(labels, smoothed_preds)
    # if pred_save_dir is not None:
    #     np.save(os.path.join(pred_save_dir, 'smoothed_pred_{}.npy'.format(target)), smoothed_preds)

    # return mse, rmse, pcc, ccc, best_window, predictions
    return smoothed_preds



test_logs = test_logs.strip().split(';')
test_logs = [i.replace(' ', '') for i in test_logs]
assert len(test_logs) == len(epoch_ids) == len(target_list)

for log, epoch, target in zip(test_logs, epoch_ids, target_list):
    # json_path = os.path.join(logs_root, log, '{}-epoch-{}_preds.json'.format(target, epoch))
    json_path = os.path.join(logs_root, log, 'epoch-{}_preds.json'.format(epoch))
    result_dir = os.path.join(result_root, log)
    mkdir(result_dir)
    result_path = os.path.join(result_dir, '{}-epoch-{}_result.txt'.format(target, epoch))
    npy_path = os.path.join(result_dir, '{}-epoch-{}_result.npy'.format(target, epoch))

    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    
    video_list = sorted(list(data_dict.keys()))
    predictions = []
    labels = []
    for video in video_list:
        predictions.append(np.array(data_dict[video]['pred']))
        labels.append(np.array(data_dict[video]['label']))

    # mse, rmse, pcc, ccc, best_window, smoothed_predictions = eval_for_val(predictions, labels, result_path=None, is_smooth=True, best_window=None)
    smoothed_predictions = eval_for_val(predictions, labels, result_path=result_path, is_smooth=True, best_window=None)
    np.save(npy_path, smoothed_predictions)
    
    