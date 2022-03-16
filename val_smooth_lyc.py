'''
验证集异步的平滑后处理：
输入一个训好的model checkpoint，给出其验证集平滑后处理后的结果
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
from utils.metrics import evaluate_regression, remove_padding, scratch_data, smooth_func, smooth_predictions
from utils.tools import calc_total_dim
# from utils.tools import get_each_dim
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
from collections import OrderedDict
import torch
import json
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

def load_config(opt_path):
    trn_opt_data = json.load(open(opt_path))
    trn_opt = OptConfig()
    trn_opt.load(trn_opt_data)
    load_dim(trn_opt)
    trn_opt.gpu_ids = opt.gpu_ids
    trn_opt.serial_batches = True # 顺序读入
    return trn_opt


def load_networks_folder(opt_config, folder_path, prefix):
    """Load all the networks from a folder.

    """
    model = create_model(opt_config)

    checkpoints = list(filter(lambda x: x.endswith('.pth'), os.listdir(folder_path)))
    for name in model.model_names:
        if isinstance(name, str):
            # load_filename = list(filter(lambda x: x.split('.')[0].endswith('net'+name), checkpoints))
            # assert len(load_filename) == 1, 'Exists file {}'.format(load_filename)
            # load_filename = load_filename[0]

            load_filename = prefix + '_net' + name + '.pth'
            load_path = os.path.join(folder_path, load_filename)

            print(load_path)##

            assert os.path.exists(load_path)

            net = getattr(model, 'net' + name)
            if isinstance(net, torch.nn.DataParallel):
                net = net.module
            print('loading the model from %s' % load_path)
            state_dict = torch.load(load_path, map_location=model.device)
            if hasattr(state_dict, '_metadata'):
                del state_dict._metadata

            net.load_state_dict(state_dict)

    model.eval()
    model.cuda()
    model.isTrain = False
    return model


# def load_model_from_checkpoint(opt_config, cpkt_dir, prefix): 
#     model = create_model(opt_config)
#     model = load_networks_folder(model, cpkt_dir, prefix)
#     model.eval()
#     model.cuda()
#     model.isTrain = False
#     return model

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
        setattr(trn_opt, "l_dim", l_dim)                # set l_dim attribute to opt


def eval_for_val(opt, model, val_dataset, target, pred_save_dir=None, is_smooth=False, best_window=None, logger=None):
    '''
    :param opt:
    :param model:
    :param val_dataset:
    :param target_idx: arousal or valence
    :param pred_save_dir
    :param is_smooth: is use the windows and smooth
    :return: post-response pairs and loss value
    '''
    model.eval()
    total_pred = []
    total_label = []
    total_length = []

    for i, data in enumerate(val_dataset):  # inner loop within one epoch
        model.set_input(data)         # unpack data from dataset and apply preprocessing
        model.test()
        lengths = data['length'].numpy()
        # pred = remove_padding(model.output.detach().cpu().numpy(), lengths)
        if model.target_name == 'both':
            pred = remove_padding(model.output[..., 0 if target == 'valence' else 1].detach().cpu().numpy(),
                                  lengths)  # [size,
        else:
            pred = remove_padding(model.output.detach().cpu().numpy(), lengths)  # [size,
        # label = remove_padding(data[opt.target].numpy(), lengths)
        label = remove_padding(data[target].numpy(), lengths)
        total_pred += pred
        total_label += label

    # calculate metrics
    if is_smooth:
        smoothed_preds, best_window = smooth_func(total_pred, total_label, best_window, logger)
    else:
        best_window = None
        smoothed_preds = total_pred
    
    smoothed_preds = scratch_data(smoothed_preds)
    total_label = scratch_data(total_label)
    mse, rmse, pcc, ccc = evaluate_regression(total_label, smoothed_preds)
    if pred_save_dir is not None:
        np.save(os.path.join(pred_save_dir, 'pred_{}.npy'.format(target)), smoothed_preds)
    # model.train()

    return mse, rmse, pcc, ccc, best_window

def get_val_logger(path, target):
    # cur_time = time.strftime('%Y-%m-%d-%H.%M.%S',time.localtime(time.time()))
    logger = logging.getLogger(__name__)
    logger.setLevel(level = logging.INFO)
    handler = logging.FileHandler(os.path.join(path, f"{target}.log"))
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    logger.addHandler(handler)
    logger.addHandler(console)
    return logger




if __name__ == '__main__':
    opt = TestOptions().parse()                        # get training options
    name = opt.name
    test_log_dir = opt.test_checkpoints
    mkdir(opt.test_log_dir)

    checkpoints = opt.test_checkpoints.strip().split(';')
    prefixs = opt.prefix_list.strip().split(';')
    print('---------------------------------')#
    print(checkpoints)#
    print('---------------------------------')#

    for checkpoint, prefix in zip(checkpoints, prefixs):
        if len(checkpoint) == 0:
            continue
        checkpoint = checkpoint.replace(' ', '')
        print('In model from {}: '.format(checkpoint))
        opt_path = os.path.join(opt.checkpoints_dir, checkpoint, 'train_opt.conf')
        trn_opt = load_config(opt_path)

        checkpoint_dir = os.path.join(opt.checkpoints_dir, checkpoint)
        val_dataset = create_dataset_with_args(trn_opt, set_name=['val'])[0]  # create a dataset given opt.dataset_mode and other options
        # model = load_model_from_checkpoint(trn_opt, checkpoint_dir, prefix)
        model = load_networks_folder(trn_opt, checkpoint_dir, prefix)

        print('Evaluating ... \n')
        npy_save_dir = os.path.join(opt.test_log_dir, 'preds', checkpoint)
        result_save_dir = os.path.join(opt.test_log_dir, 'results', checkpoint)
        result_save_path = os.path.join(result_save_dir, 'result_{}.txt'.format(opt.test_target))
        logger_save_dir = os.path.join(opt.test_log_dir, 'logs', checkpoint)   # get logger path

        mkdir(npy_save_dir)
        mkdir(result_save_dir)
        mkdir(logger_save_dir)

        logger = get_val_logger(logger_save_dir, opt.test_target)            # get logger
        mse, rmse, pcc, ccc, best_window = eval_for_val(trn_opt, model,val_dataset, opt.test_target, pred_save_dir=npy_save_dir, is_smooth=True, logger=logger)
        metrics_smoothed = {'mse': mse, 'rmse': rmse, 'pcc': pcc, 'ccc': ccc, 'best_window': best_window}
        with open(result_save_path, 'w') as f:
            for key in metrics_smoothed.keys():
                line = key + ':\t' + str(metrics_smoothed[key]) + '\n'
                f.write(line)

        # logging.shutdown() # 在每次跑完一个模型后及时关闭logger，防止后续循环中不断在前面的logging文件中续写。
