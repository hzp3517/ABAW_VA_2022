import os
import numpy as np, argparse, time, pickle, random, json

from opts.train_opts import TrainOptions
from data import create_dataset, create_dataset_with_args
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


def eval(model, val_iter, target='valence', smooth=False):  # target = valence, arousal
    model.eval()
    total_pred = []
    total_label = []
    total_length = []

    results = {}
    for i, data in enumerate(val_iter):  # inner loop within one epoch
        model.set_input(data)  # unpack data from dataset and apply preprocessing
        model.test()
        lengths = data['length'].numpy()
        if model.target_name == 'both':
            pred = remove_padding(model.output[..., 0 if target == 'valence' else 1].detach().cpu().numpy(),
                                  lengths)  # [size,
        else:
            pred = remove_padding(model.output.detach().cpu().numpy(), lengths)  # [size,
        label = remove_padding(data[target].numpy(), lengths)
        total_pred += pred
        total_label += label
        for j, video_id in enumerate(data['video_id']):
            results[video_id] = {'video_id': video_id,
                                 'pred': pred[j].tolist(),
                                 'label': label[j].tolist()}

    # calculate metrics
    best_window = None
    if smooth:
        total_pred, best_window = smooth_func(total_pred, total_label, best_window=best_window, logger=logger)

    total_pred = scratch_data(total_pred)
    total_label = scratch_data(total_label)
    mse, rmse, pcc, ccc = evaluate_regression(total_label, total_pred)
    model.train()

    return mse, rmse, pcc, ccc, best_window, results


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

    dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options
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
            mse, rmse, pcc, ccc, window, _ = eval(model, dataset, target=target)
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



            # if best_ccc.get(target) is None or best_ccc[target] < ccc:
            #     best_ccc[target] = ccc
            #     if not os.path.exists(os.path.join(opt.checkpoints_dir, opt.name, 'seed_' + str(seed))):
            #         os.makedirs(os.path.join(opt.checkpoints_dir, opt.name, 'seed_' + str(seed)))
            #     if opt.save_model:  # cache our model every <save_epoch_freq> epochs
            #         logger.info('saving the model of best %s at the end of epoch %d, iters %d' % (target, epoch, total_iters))
            #         # model.save_networks('latest')
            #         model.save_networks(os.path.join('seed_' + str(seed), 'best_' + target))

            #         pred_save_path = os.path.join(opt.checkpoints_dir, opt.name, 'seed_{}'.format(seed), 'best_pred_{}.json'.format(target))
            #         json.dump(preds, open(pred_save_path, 'w'))

    # print best eval result
    # if opt.target == 'both':
    #     target_set.append('average')
    #     metrics['average'] = [[(a + b) / 2 for a, b in zip(x, y)] for x, y in
    #                           zip(metrics['arousal'], metrics['valence'])]

    # result_path = os.path.join(logger_path, 'result.txt')
    # with open(result_path, 'a') as f:
        # f.write('Using Seed: {}\n'.format(seed))
    
    best_epoch_list = []
    for target in target_set:
        logger.info('Best eval epoch %d found with ccc %f on %s' % (best_eval_epoch[target], best_eval_ccc[target], target))
        logger.info(opt.name)
        result_save_path = os.path.join(opt.log_dir, opt.name, 'epoch-{}_preds.json'.format(best_eval_epoch[target]))
        json.dump(best_eval_result[target], open(result_save_path, 'w'))
        best_epoch_list.append(best_eval_epoch[target])
        
    clean_chekpoints(opt.checkpoints_dir, opt.name, best_epoch_list)
