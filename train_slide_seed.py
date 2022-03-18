import os
import numpy as np, argparse, time, pickle, random, json

from opts.train_opts import TrainOptions
from data import create_dataset, create_dataset_with_args, CustomDatasetDataLoader
from models import create_model
from utils.logger import get_logger
from utils.path import make_path
from utils.metrics import evaluate_regression, remove_padding, scratch_data, smooth_func
from utils.tools import calc_total_dim
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import torch
from collections import OrderedDict
import fcntl
import csv
from torch.utils.tensorboard import SummaryWriter

import warnings
warnings.filterwarnings("ignore")

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def test(model, tst_iter):
    pass


def get_avg_result(val_iter, seg_videoId_lst, total_pred, total_label):
    '''
    total_pred: (total_num_segs, win_len, reg_output)
    total_label: (total_num_segs, win_len, reg_output)
    '''
    video_list = val_iter.dataset.video_list
    video_len_list = val_iter.dataset.video_len_list
    win_len = val_iter.dataset.win_len
    hop_len = val_iter.dataset.hop_len

    final_pred = []
    final_label = []
    all_video_dict = {}

    cur_video = video_list[0]
    video_pred = np.zeros(video_len_list[0])
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
        label = total_label[idx]

        # for seg_videoId, pred, label in zip(seg_videoId_lst, total_pred, total_label):
        if seg_videoId == cur_video:
            assert start < len(video_pred)
            if end <= len(video_pred):
                video_pred[start: end] = pred
                video_label[start: end] = label
                video_cnt_lst[start: end] = video_cnt_lst[start: end] + 1
            else:
                pad_num = end - len(video_pred)
                video_pred[start:] = pred[:-pad_num]
                video_label[start:] = label[:-pad_num]
                video_cnt_lst[start:] = video_cnt_lst[start:] + 1
            start += hop_len
            end += hop_len
            idx += 1
        else:
            # 把上一个视频处理好
            assert np.all(video_cnt_lst)
            video_pred = video_pred / video_cnt_lst
            video_label = video_label / video_cnt_lst
            video_dict = {'video_id': cur_video, 'pred': video_pred.tolist(), 'label': video_label.tolist()}
            final_pred.append(video_pred)
            final_label.append(video_label)
            all_video_dict[cur_video] = video_dict

            # 准备好处理下一个视频
            cur_video = seg_videoId
            video_cnt += 1
            video_pred = np.zeros(video_len_list[video_cnt])
            video_label = np.zeros(video_len_list[video_cnt])
            video_cnt_lst = np.zeros(video_len_list[video_cnt])
            start = 0
            end = win_len

    # 处理好最后一个视频
    assert np.all(video_cnt_lst)
    video_pred = video_pred / video_cnt_lst
    video_label = video_label / video_cnt_lst
    video_dict = {'video_id': cur_video, 'pred': video_pred.tolist(), 'label': video_label.tolist()}
    final_pred.append(video_pred)
    final_label.append(video_label)
    all_video_dict[cur_video] = video_dict
    
    assert len(final_pred) == len(video_list)

    return final_pred, final_label, all_video_dict


def eval(model, val_iter, target='valence', smooth=False):  # target = valence, arousal
    model.eval()
    # total_data = []
    seg_videoId_lst = []
    total_pred = []
    total_label = []

    # results = {}
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        if model.target_name == 'both':
            pred = model.output[..., 0 if target == 'valence' else 1].detach().cpu().numpy()
        else:
            pred = model.output.detach().cpu().squeeze(0).numpy()
        label = data[target].numpy()

        # total_data.append(data)
        seg_videoId_lst += data['video_id']
        total_pred.append(pred) # pred: (bs, win_len, reg_output)
        total_label.append(label)
    total_pred = np.concatenate(total_pred, axis=0) # total_pred: (total_num_segs, win_len, reg_output)
    total_label = np.concatenate(total_label, axis=0) # total_pred: (total_num_segs, win_len, reg_output)

    final_pred, final_label, final_results = get_avg_result(val_iter, seg_videoId_lst, total_pred, total_label)

    final_pred = np.array(final_pred)
    final_label = np.array(final_label)

    # calculate metrics
    best_window = None
    if smooth:
        final_pred, best_window = smooth_func(final_pred, final_label, best_window=best_window, logger=logger)

    final_pred = scratch_data(final_pred)
    final_label = scratch_data(final_label)
    mse, rmse, pcc, ccc = evaluate_regression(final_label, final_pred)
    model.train()

    return mse, rmse, pcc, ccc, best_window, final_results


def clean_chekpoints(checkpoints_dir, expr_name, store_epoch_list):
    root = os.path.join(checkpoints_dir, expr_name)
    # if not checkpoint.startswith(str(store_epoch) + '_') and checkpoint.endswith('pth'):
    for checkpoint in os.listdir(root):
        isStoreEpoch = False
        for store_epoch in store_epoch_list:
            if checkpoint.startswith(str(store_epoch) + '_'):
                isStoreEpoch = True
                break
        if not isStoreEpoch and checkpoint.endswith('pth'):
            os.remove(os.path.join(root, checkpoint))



if __name__ == '__main__':
    best_window = None
    opt = TrainOptions().parse()  # get training options

    seed = 99 + opt.run_idx
    seed_everything(seed)

    if not os.path.exists(opt.log_dir):
        os.makedirs(opt.log_dir)
    logger_path = os.path.join(opt.log_dir, opt.name)  # get logger path
    suffix = opt.name  # get logger suffix
    logger = get_logger(logger_path, suffix)            # get logger
    logger.info('Using seed: {}'.format(seed))

    dataset, val_dataset, train_eval_dataset = create_dataset_with_args(opt, set_name=['train', 'val', 'train_eval'])  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)  # get the number of images in the dataset.
    logger.info('The number of training samples = %d' % dataset_size)
    # calculate input dims
    if opt.feature_set != 'None':
        input_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.feature_set.split(','))))
        setattr(opt, "input_dim", input_dim)  # set input_dim attribute to opt
    if hasattr(opt, "a_features"):
        a_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.a_features.split(','))))
        setattr(opt, "a_dim", a_dim)  # set a_dim attribute to opt
    if hasattr(opt, "v_features"):
        v_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.v_features.split(','))))
        setattr(opt, "v_dim", v_dim)  # set v_dim attribute to opt
    if hasattr(opt, "l_features"):
        l_dim = calc_total_dim(list(map(lambda x: x.strip(), opt.l_features.split(','))))
        setattr(opt, "l_dim", l_dim)  # set l_dim attribute to opt

    model = create_model(opt, logger=logger)  # create a model given opt.model and other options
    model.setup(opt)  # regular setup: load and print networks; create schedulers
    total_iters = 0  # the total number of training iterations
    # best_eval_ccc = 0                           # record the best eval UAR
    # best_eval_epoch = -1                        # record the best eval epoch
    # best_eval_window = None
    # writer = SummaryWriter(logger_path)

    target_set = ['valence', 'arousal'] if opt.target == 'both' else [opt.target]
    metrics = {}
    best_eval_ccc = {}
    best_eval_epoch = {}
    best_eval_window = {}
    best_eval_result = {}
    
    for target in target_set:
        metrics[target] = []
        best_eval_ccc[target] = 0
        best_eval_epoch[target] = -1
        best_eval_window[target] = None
        best_eval_result[target] = None

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):  # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()  # timer for data loading per iteration
        epoch_iter = 0  # the number of training iterations in current epoch, reset to 0 every epoch
        batch_count = 0  # batch个数统计，用于粗略计算整个epoch的loss
        iter_data_statis = 0.0  # record total data reading time
        cur_epoch_losses = OrderedDict()
        for name in model.loss_names:
            cur_epoch_losses[name] = 0
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time = time.time()  # timer for computation per iteration
            iter_data_statis += iter_start_time - iter_data_time
            total_iters += 1  # opt.batch_size
            epoch_iter += opt.batch_size
            batch_count += 1
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.run()  # calculate loss functions, get gradients, update network weights

            # ---------在每个batch都获取一次loss，并加入cur_epoch_losses-------------
            losses = model.get_current_losses()
            for name in losses.keys():
                cur_epoch_losses[name] += losses[name]
            # ---------------------------------------------------------------------

            if total_iters % opt.print_freq == 0:  # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
                            ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**losses))
                
            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                logger.info('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            logger.info('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        logger.info('End of training epoch %d / %d \t Time Taken: %d sec, Data loading: %d sec' % (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time, iter_data_statis))
        model.update_learning_rate()  # update learning rates at the end of every epoch.

        # -----得到并打印当前epoch的loss------
        for name in cur_epoch_losses:
            cur_epoch_losses[name] /= batch_count # 这样直接对各个batch内的平均loss取平均的方法并非完全精确，因为最后一个batch内数据的数量可能少于batch_size，但应该也不会差太多。
        logger.info('Cur epoch {}'.format(epoch) + ' loss ' +
                    ' '.join(map(lambda x: '{}:{{{}:.4f}}'.format(x, x), model.loss_names)).format(**cur_epoch_losses))
        # -----------------------------------

        # # ---tensorboard---
        # writer = SummaryWriter(os.path.join(logger_path, 'tensorboard'))
        # for name in cur_epoch_losses:
        #    writer.add_scalar("Loss_{}/train".format(name), cur_epoch_losses[name], epoch)
        # # -----------------

        # eval train set
        for target in target_set:
            mse, rmse, pcc, ccc, window, _ = eval(model, train_eval_dataset, target=target)
            logger.info('Train result on %s of epoch %d / %d mse %.4f rmse %.4f pcc %.4f ccc %.4f' % (target, epoch, opt.niter + opt.niter_decay, mse, rmse, pcc, ccc))

            # eval val set
            mse, rmse, pcc, ccc, window, preds = eval(model, val_dataset, target=target)
            logger.info('Val result on %s of epoch %d / %d mse %.4f rmse %.4f pcc %.4f ccc %.4f' % (target, epoch, opt.niter + opt.niter_decay, mse, rmse, pcc, ccc))
            metrics[target].append((mse, rmse, pcc, ccc))

            if ccc > best_eval_ccc[target]:
                best_eval_epoch[target] = epoch
                best_eval_ccc[target] = ccc
                best_eval_window[target] = window
                best_eval_result[target] = preds
    
    best_epoch_list = []
    for target in target_set:
        logger.info('Best eval epoch %d found with ccc %f on %s' % (best_eval_epoch[target], best_eval_ccc[target], target))
        logger.info(opt.name)
        result_save_path = os.path.join(opt.log_dir, opt.name, '{}-epoch-{}_preds.json'.format(target, best_eval_epoch[target]))
        json.dump(best_eval_result[target], open(result_save_path, 'w'))
        best_epoch_list.append(best_eval_epoch[target])
        
    clean_chekpoints(opt.checkpoints_dir, opt.name, best_epoch_list)