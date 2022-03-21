'''
val_pred_{valence/arousal}_nosmooth_ori.json
val_pred_{valence/arousal}_nosmooth.json 
val_pred_{valence/arousal}_smooth.json 
val_pred_{valence/arousal}_result.txt
tst_pred_{valence/arousal}_smooth.json

smooth window: 
- for valence, win_size = 50;
- for arousal, win_size = 20.
'''

import os
import time
import datetime
import numpy as np
from opts.test_opts import TestOptions
from data import create_dataset, create_dataset_with_args
from models import create_model
from utils.logger import get_logger
from utils.path import make_path
from utils.metrics import evaluate_regression, remove_padding, scratch_data, smooth_func
from utils.tools import calc_total_dim
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from collections import OrderedDict
import torch
import json
import sys
from data import find_dataset_using_name
import fcntl
import csv
from models.model_utils.config import OptConfig

def mkdir(path):
    try:
        os.makedirs(path)
    except FileExistsError:
        pass

def load_config(opt_path, test_opt):
    trn_opt_data = json.load(open(opt_path))
    trn_opt = OptConfig()
    trn_opt.load(trn_opt_data)
    load_dim(trn_opt)
    trn_opt.gpu_ids = test_opt.gpu_ids
    trn_opt.serial_batches = True # 顺序读入
    return trn_opt


def load_networks_folder_with_prefix(model, folder_path, prefix):
    """Load all the networks from a folder.
    - folder_path: xxx/checkpoints/.../[prefix]
        (the model path is: [prefix]_net_xx.pth)
    """
    # prefix = folder_path.split('/')[-1]
    # folder_path = '/'.join(folder_path.split('/')[:-1])
    prefix = prefix + '_'
    checkpoints = list(filter(lambda x: x.startswith(prefix) and x.endswith('.pth'), os.listdir(folder_path)))
    for name in model.model_names:
        if isinstance(name, str):
            load_filename = list(filter(lambda x: x.split('.')[0].endswith('net'+name), checkpoints))
            assert len(load_filename) == 1, 'Exists file {}'.format(load_filename)
            load_filename = load_filename[0]
            load_path = os.path.join(folder_path, load_filename)
            net = getattr(model, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location=model.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net.load_state_dict(state_dict)

    return model


def load_model_from_checkpoint(opt_config, cpkt_dir, prefix):
    model = create_model(opt_config)
    model = load_networks_folder_with_prefix(model, cpkt_dir, prefix)
    model.eval()
    model.cuda()
    model.isTrain = False
    return model

def load_dim(trn_opt):
    if trn_opt.feature_set != 'None':
        input_dim = calc_total_dim(list(map(lambda x: x.strip(), trn_opt.feature_set.split(','))))
        setattr(trn_opt, "input_dim", input_dim)                # set input_dim attribute to opt
    if hasattr(trn_opt, "a_features"):
        a_dim = calc_total_dim(list(map(lambda x: x.strip(), trn_opt.a_features.split(','))))
        setattr(trn_opt, "a_dim", a_dim)                # set a_dim attribute to opt
    if hasattr(trn_opt, "v_features"):
        v_dim = calc_total_dim(list(map(lambda x: x.strip(), trn_opt.v_features.split(','))))
        setattr(trn_opt, "v_dim", v_dim)                # set v_dim attribute to opt
    if hasattr(trn_opt, "l_features"):
        l_dim = calc_total_dim(list(map(lambda x: x.strip(), trn_opt.l_features.split(','))))
        setattr(trn_opt, "l_dim", l_dim)    


def process_preds(ori_pred):
    pred = ori_pred.copy()
    too_high = pred > 1
    too_low = pred < -1
    pred[too_high] = 1
    pred[too_low] = -1
    return pred


def smooth_prediction(pred, window, mean_or_binomial=True):
    """
    Args:
      pred
      window: int (one side timesteps to average)
    Return:
      smoothed_pred: shape=pred
    """
    if window == 0:
        return pred
    if not mean_or_binomial:
        from scipy.stats import binom
        binomial_weights = np.zeros((window * 2 + 1,))
        for i in range(window + 1):
            binomial_weights[i] = binomial_weights[-i - 1] = binom.pmf(i, window * 2, 0.5)
    # smoothed_preds = []
    # for pred in preds:
    smoothed_pred = []
    for t in range(pred.shape[0]):
        left = np.max([0, t - window])
        right = np.min([pred.shape[0], t + window])
        if mean_or_binomial:
            smoothed_pred.append(np.mean(pred[left: right + 1]))
        else:
            if left <= 0:
                weights = binomial_weights[window - t:]
            elif right >= pred.shape[0]:
                weights = binomial_weights[:pred.shape[0] - t - window - 1]
            else:
                weights = binomial_weights
            smoothed_pred.append(np.sum(pred[left: right + 1] * weights))
    # smoothed_preds.append(np.array(smoothed_pred))
    # smoothed_preds = np.array(smoothed_preds, dtype=object)
    # return smoothed_preds
    smoothed_pred = np.array(smoothed_pred)
    return smoothed_pred


def get_avg_result(data_iter, seg_videoId_lst, total_pred, total_label=None):
    '''
    total_pred: (total_num_segs, win_len, reg_output)
    total_label: (total_num_segs, win_len, reg_output)
    '''
    video_list = data_iter.dataset.video_list
    video_len_list = data_iter.dataset.video_len_list
    win_len = data_iter.dataset.win_len
    hop_len = data_iter.dataset.hop_len

    final_pred = []
    final_label = []
    all_video_dict = {}

    cur_video = video_list[0]
    video_pred = np.zeros(video_len_list[0])
    if total_label:
        video_label = np.zeros(video_len_list[0])
    video_cnt_lst = np.zeros(video_len_list[0])
    start = 0
    end = win_len

    idx = 0
    video_cnt = 0
    total_seg_num = len(seg_videoId_lst)
    while idx < total_seg_num:
        seg_videoId = seg_videoId_lst[idx]
        pred = total_pred[idx]
        if total_label:
            label = total_label[idx]

        # for seg_videoId, pred, label in zip(seg_videoId_lst, total_pred, total_label):
        if seg_videoId == cur_video:
            assert start < len(video_pred)
            if end <= len(video_pred):
                video_pred[start: end] = video_pred[start: end] + pred
                if total_label:
                    video_label[start: end] = video_label[start: end] + label
                video_cnt_lst[start: end] = video_cnt_lst[start: end] + 1
            else:
                pad_num = end - len(video_pred)
                video_pred[start:] = video_pred[start:] + pred[:-pad_num]
                if total_label:
                    video_label[start:] = video_label[start:] + label[:-pad_num]
                video_cnt_lst[start:] = video_cnt_lst[start:] + 1
            start += hop_len
            end += hop_len
            idx += 1
        else:
            # 把上一个视频处理好
            assert np.all(video_cnt_lst)
            video_pred = video_pred / video_cnt_lst
            final_pred.append(video_pred)
            if total_label:
                video_label = video_label / video_cnt_lst
                video_dict = {'video_id': cur_video, 'pred': video_pred, 'label': video_label}
                final_label.append(video_label)
            else:
                video_dict = {'video_id': cur_video, 'pred': video_pred}
            all_video_dict[cur_video] = video_dict

            # 准备好处理下一个视频
            cur_video = seg_videoId
            video_cnt += 1
            video_pred = np.zeros(video_len_list[video_cnt])
            if total_label:
                video_label = np.zeros(video_len_list[video_cnt])
            video_cnt_lst = np.zeros(video_len_list[video_cnt])
            start = 0
            end = win_len

    # 处理好最后一个视频
    assert np.all(video_cnt_lst)
    video_pred = video_pred / video_cnt_lst
    final_pred.append(video_pred)
    if total_label:
        video_label = video_label / video_cnt_lst
        video_dict = {'video_id': cur_video, 'pred': video_pred, 'label': video_label}
        final_label.append(video_label)
    else:
        video_dict = {'video_id': cur_video, 'pred': video_pred}
    all_video_dict[cur_video] = video_dict

    assert len(final_pred) == len(video_list)

    if total_label:
        return final_pred, final_label, all_video_dict
    else:
        return final_pred, all_video_dict


def eval_for_val(model, val_dataset, target, best_window):
    '''
    :param model:
    :param val_dataset:
    :param target: arousal or valence
    :param best_window: set the window size for smoothing
    return:
    - nosmooth_ori_dict
    - nosmooth_ori_ccc
    - nosmooth_dict
    - nosmooth_ccc
    - smooth_dict
    - smooth_ccc
    - window
    '''
    model.eval()
    seg_videoId_lst = []
    total_pred = []
    total_label = []

    nosmooth_ori_dict = {}
    nosmooth_dict = {}
    smooth_dict = {}

    for i, data in enumerate(val_dataset):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        if model.target_name == 'both':
            pred = model.output[..., 0 if target == 'valence' else 1].detach().cpu().numpy()
        else:
            pred = model.output.detach().cpu().squeeze(0).numpy()
        label = data[target].numpy()

        seg_videoId_lst += data['video_id']
        total_pred.append(pred) # pred: (bs, win_len, reg_output)
        total_label.append(label)

    total_pred = np.concatenate(total_pred, axis=0) # total_pred: (total_num_segs, win_len, reg_output)
    total_label = np.concatenate(total_label, axis=0) # total_pred: (total_num_segs, win_len, reg_output)

    avg_pred, avg_label, avg_results = get_avg_result(val_dataset, seg_videoId_lst, total_pred, total_label)

    avg_pred = np.array(avg_pred)
    avg_label = np.array(avg_label)

    for video_id in list(avg_results.keys()):
        ori_pred = avg_results[video_id]['pred']
        label = avg_results[video_id]['label']
        processed_pred = process_preds(ori_pred)
        smooth_pred = smooth_prediction(ori_pred, best_window)
        smooth_pred = process_preds(smooth_pred)

        nosmooth_ori_dict[video_id] = {'video_id': video_id,
                                        'pred': ori_pred.tolist(),
                                        'label': label.tolist()}
        nosmooth_dict[video_id] = {'video_id': video_id,
                                        'pred': processed_pred.tolist(),
                                        'label': label.tolist()}
        smooth_dict[video_id] = {'video_id': video_id,
                                        'pred': smooth_pred.tolist(),
                                        'label': label.tolist()}

    nosmooth_ori_pred = scratch_data(avg_pred)
    smoothed_pred, best_window = smooth_func(avg_pred, avg_label, best_window)

    smoothed_pred = scratch_data(smoothed_pred)
    avg_label = scratch_data(avg_label)

    nosmooth_pred = process_preds(nosmooth_ori_pred)
    smoothed_pred = process_preds(smoothed_pred)

    nosmooth_ori_mse, nosmooth_ori_rmse, nosmooth_ori_pcc, nosmooth_ori_ccc = evaluate_regression(avg_label, nosmooth_ori_pred)
    nosmooth_mse, nosmooth_rmse, nosmooth_pcc, nosmooth_ccc = evaluate_regression(avg_label, nosmooth_pred)
    smoothed_mse, smoothed_rmse, smoothed_pcc, smoothed_ccc = evaluate_regression(avg_label, smoothed_pred)

    return nosmooth_ori_dict, nosmooth_ori_ccc, nosmooth_dict, nosmooth_ccc, smooth_dict, smoothed_ccc, best_window


def get_test_pred(model, test_dataset, target, best_window):
    model.eval()
    seg_videoId_lst = []
    total_pred = []
    smooth_dict = {}

    for i, data in enumerate(test_dataset):  # inner loop within one epoch
        model.set_input(data, load_label=False)         # unpack data from dataset and apply preprocessing
        model.test()
        if model.target_name == 'both':
            pred = model.output[..., 0 if target == 'valence' else 1].detach().cpu().numpy()
        else:
            pred = model.output.detach().cpu().squeeze(0).numpy()

        seg_videoId_lst += data['video_id']
        total_pred.append(pred) # pred: (bs, win_len, reg_output)

    total_pred = np.concatenate(total_pred, axis=0) # total_pred: (total_num_segs, win_len, reg_output)

    avg_pred, avg_results = get_avg_result(test_dataset, seg_videoId_lst, total_pred)

    for video_id in list(avg_results.keys()):
        ori_pred = avg_results[video_id]['pred']
        smooth_pred = smooth_prediction(ori_pred, best_window)
        smooth_pred = process_preds(smooth_pred)
        smooth_dict[video_id] = {'video_id': video_id,
                                        'pred': smooth_pred.tolist()}

    return smooth_dict, best_window


if __name__ == '__main__':
    opt = TestOptions().parse()                        # get training options
    name = opt.name
    test_log_dir = opt.test_checkpoints
    mkdir(opt.test_log_dir)

    checkpoints = opt.test_checkpoints.strip().split(';')
    print('---------------------------------')#
    print(checkpoints)#
    print('---------------------------------')#

    for checkpoint in checkpoints:
        prefix = checkpoint.split('/')[-1]
        checkpoint = '/'.join(checkpoint.split('/')[:-1])

        if len(checkpoint) == 0:
            continue
        checkpoint = checkpoint.replace(' ', '')
        print('In model from {}: '.format(checkpoint))
        opt_path = os.path.join(opt.checkpoints_dir, checkpoint, 'train_opt.conf')
        trn_opt = load_config(opt_path, opt)
        
        checkpoint_dir = os.path.join(opt.checkpoints_dir, checkpoint)
        val_dataset, test_dataset = create_dataset_with_args(trn_opt, set_name=['val', 'test'])  # create a dataset given opt.dataset_mode and other options
        model = load_model_from_checkpoint(trn_opt, checkpoint_dir, prefix)

        save_dir = os.path.join(opt.test_log_dir, checkpoint)
        mkdir(save_dir)
        val_nosmooth_ori_pred_path = os.path.join(save_dir, 'val_pred_{}_nosmooth_ori.json'.format(opt.test_target))
        val_nosmooth_pred_path = os.path.join(save_dir, 'val_pred_{}_nosmooth.json'.format(opt.test_target))
        val_smooth_pred_path = os.path.join(save_dir, 'val_pred_{}_smooth.json'.format(opt.test_target))
        val_result_path = os.path.join(save_dir, 'val_pred_{}_result.txt'.format(opt.test_target))
        test_smooth_pred_path = os.path.join(save_dir, 'tst_pred_{}_smooth.json'.format(opt.test_target))
        assert opt.test_target in ['valence', 'arousal']
        
        setting_window = 50 if opt.test_target == 'valence' else 20

        print('Evaluating on the val set ... \n')
        nosmooth_ori_dict, nosmooth_ori_ccc, nosmooth_dict, nosmooth_ccc, smoothed_dict, smoothed_ccc, window = eval_for_val(model, val_dataset, opt.test_target, setting_window)
        json.dump(nosmooth_ori_dict, open(val_nosmooth_ori_pred_path, 'w'))
        json.dump(nosmooth_dict, open(val_nosmooth_pred_path, 'w'))
        json.dump(smoothed_dict, open(val_smooth_pred_path, 'w'))
        with open(val_result_path, 'w') as f:
            line = []
            line += 'nosmooth_ori_ccc:\t{}\n'.format(nosmooth_ori_ccc)
            line += 'nosmooth_ccc:\t{}\n'.format(nosmooth_ccc)
            line += 'smoothed_ccc:\t{}\n'.format(smoothed_ccc)
            line += 'setting window:\t{}'.format(window)
            f.writelines(line)

        print('Getting the predictions of the test set ... \n')
        test_smoothed_dict, window = get_test_pred(model, test_dataset, opt.test_target, setting_window)
        json.dump(test_smoothed_dict, open(test_smooth_pred_path, 'w'))