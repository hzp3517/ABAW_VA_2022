'''
利用预训练的wav2vec模型，在本数据集上做VA回归任务
'''
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2Model
from torch.nn.utils.rnn import pad_sequence
import math

import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code/')#
from models.base_model import BaseModel#
from models.networks.regressor import FcRegressor#

class Wav2vecModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--regress_layers', default='256', type=str,
                            help='size of classifier hidden size, split by comma')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate')
        parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1'], help='use MSEloss or L1loss')
        parser.add_argument('--task', type=str, default='v+a', choices=['v', 'a', 'v+a'], help='the task target.')
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the BaselineModel class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = [opt.loss_type] # get_current_losses: getattr(self, 'loss_' + name)
        self.regress_layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.dropout_rate = opt.dropout_rate
        self.task = opt.task
        self.label_fps = 30 # 标签频率每秒30帧
        self.frame_rate = 0.02 # wav2vec 0.02s一帧
        self.model_names = ['_wav2vec', '_reg']
        self.pretrained_model = ['_wav2vec']
        self.net_wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

        if self.task == 'v' or self.task == 'a':
            self.net_reg = FcRegressor(768, self.regress_layers, 1, dropout=self.dropout_rate)
        else:
            self.net_reg = FcRegressor(768, self.regress_layers, 2, dropout=self.dropout_rate)

        #settings
        if self.isTrain:
            if opt.loss_type == 'mse':
                self.criterion_reg = torch.nn.MSELoss(reduction='sum')
            else:
                self.criterion_reg = torch.nn.L1Loss(reduction='sum')
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def set_input(self, input, load_label=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.len_speech = input['len_speech'].to(self.device) # (bs,)
        self.speech = input['speech'].to(self.device) # (bs, len_speech)
        if load_label:
            self.valid = input['valid'].to(self.device) # (bs, max_clip_length*30)
            if self.task == 'v':
                self.target = input['label'].permute(2, 0, 1)[0].to(self.device) # (bs, max_clip_length*30)
            elif self.task == 'a':
                self.target = input['label'].permute(2, 0, 1)[1].to(self.device) # (bs, max_clip_length*30)
            else:
                self.target = input['label'].to(self.device) # (bs, max_clip_length*30, 2)

    def run(self):
        """After feed a batch of samples, Run the model."""
        # forward
        prediction = self.forward_step() # (bs, aligned_len_speech, 2) or (bs, aligned_len_speech, 1)
        if self.task == 'v' or self.task == 'a':
            prediction = prediction.squeeze(dim=-1) # (bs, aligned_len_speech)
        
        # backward
        if self.isTrain:
            self.optimizer.zero_grad()
            self.backward_step(prediction, self.target, self.valid)
            self.optimizer.step()
        
        self.output = prediction


    def align_timestamp(self, batch_ft, num_frames):
        '''
        The wav2vec has 50 fts per second. There are 30 labels per second.
        We need to align the feature frequency to the label frequency.
        '''
        start_time_list = [i * (1.0/self.label_fps) for i in range(num_frames)]
        end_time_list = [((1.0/self.label_fps) + i * (1.0/self.label_fps)) for i in range(num_frames)]

        # 把每个时刻的值精度缩小一下，防止因为float本身有些小数最后一位不够精确导致多算一条frame
        start_time_list = [round(i, 4) for i in start_time_list]
        end_time_list = [round(i, 4) for i in end_time_list]

        # 确保 batch_ft pad到足够长
        finish_idx = math.ceil(end_time_list[-1] / self.frame_rate)
        if batch_ft.shape[1] < finish_idx:
            pad_num = finish_idx + 1 - batch_ft.shape[0]
            pad_ft = batch_ft[:, -1, :] # (bs, dim)
            pad_ft = pad_ft.unsqueeze(1).repeat(1, pad_num, 1) # (bs, pad_num, dim)
            batch_ft = torch.cat((batch_ft, pad_ft), dim=1)
        
        segment_ft = []
        for start, end in zip(start_time_list, end_time_list):
            start_idx = math.floor(round(start / self.frame_rate, 4)) # 同样也需要在这里缩小一下精度
            end_idx = math.ceil(round(end / self.frame_rate, 4))
            frame_ft = batch_ft[:, start_idx: end_idx, :]
            frame_ft = frame_ft.mean(dim=1)
            segment_ft.append(frame_ft)
        segment_ft = torch.stack(segment_ft, dim=1)
        assert segment_ft.shape[1] == num_frames

        return segment_ft # (bs, num_frames, dim)



    def forward_step(self):
        batch_ft = self.net_wav2vec(self.speech).last_hidden_state # (bs, len_speech, dim=768)
        aligned_len = self.target.shape[1]
        batch_ft = self.align_timestamp(batch_ft, aligned_len) # (bs, num_frames, dim=768)

        pred, _ = self.net_reg(batch_ft) # (bs, aligned_len_speech, 2) or (bs, aligned_len_speech, 1)
        return pred

    def backward_step(self, pred, target, mask):
        """Calculate the loss for back propagation"""
        if self.task == 'v+a':
            mask = mask.unsqueeze(2).repeat(1, 1, 2)
        
        pred = pred * mask
        target = target * mask
        valid_data_num = torch.sum(mask)
        self.loss_mse = self.criterion_reg(pred, target) / valid_data_num
        self.loss_mse.backward(retain_graph=False)


if __name__ == '__main__':
    import sys
    sys.path.append('/data8/hzp/ABAW_Expression_2022/code/utils')#
    from data import create_dataset, create_dataset_with_args

    class test:
        feature_set = 'None'
        input_dim = None
        lr = 1e-4
        beta1 = 0.5
        batch_size = 2
        epoch_count = 1
        niter = 20
        niter_decay=30
        gpu_ids = [1]
        isTrain = True
        checkpoints_dir = ''
        name = ''
        cuda_benchmark = ''
        max_read_num = 10
        max_clip_length = 10
        dataset_mode = 'audio_clip_toy'
        num_threads = 0
        max_dataset_size = float("inf")
        serial_batches = True
        lr_policy = ''
        init_type = 'kaiming'
        init_gain = 0.02
        verbose = ''

        regress_layers = '256'
        dropout_rate = 0.3
        loss_type = 'mse'
        task = 'v'

    opt = test()
    net_a = Wav2vecModel(opt)
    
    # dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options
    dataset = create_dataset_with_args(opt, set_name=['train'])[0]  # create a dataset given opt.dataset_mode and other options

    net_a.setup(opt)                            # regular setup: load and print networks; create schedulers

    total_iters = 0                             # the total number of training iterations
    
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch            
        for i, data in enumerate(dataset, 1):  # inner loop within one epoch
            total_iters += 1                # opt.batch_size
            epoch_iter += opt.batch_size
            net_a.set_input(data)           # unpack data from dataset and apply preprocessing
            net_a.run()
            
            print(net_a.output)
            print(net_a.output.shape)


    # for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
    #     epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch            
    #     for i, data in enumerate(val_dataset, 1):  # inner loop within one epoch
    #         total_iters += 1                # opt.batch_size
    #         epoch_iter += opt.batch_size
    #         net_a.set_input(data)           # unpack data from dataset and apply preprocessing
    #         net_a.run()
            
    #         print(net_a.output)
    #         print(net_a.output.shape) # (1, 2)
