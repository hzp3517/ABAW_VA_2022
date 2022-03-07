import torch
import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
# from .base_model import BaseModel
# from .networks.tcn import TCNModel
# from .networks.regressor import FcRegressor
# from .networks.loss import PCCLoss

import sys
sys.path.append('/data8/hzp/ABAW_VA_2022/code')#
from models.base_model import BaseModel#
from models.networks.tcn import TCNModel#
from models.networks.regressor import FcRegressor#

class TcnModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of lstm')
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        # parser.add_argument('--hidden_size', default=256, type=int, help='lstm hidden layer')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        parser.add_argument('--tcn_dropout', default=0.2, type=float, help='drop out rate of TCN layers')
        parser.add_argument('--tcn_channels', default='512', help='number of TCN channels, split by the comma.')
        parser.add_argument('--kernel_size', type=int, default=3, help='kernel size of TCN')
        parser.add_argument('--target', default='arousal', type=str, help='one of [arousal, valence]')
        parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1', 'ccc'], help='use MSEloss or L1loss or CCCloss')
        return parser
    
    def __init__(self, opt, logger=None):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = ['MSE']
        self.model_names = ['_seq', '_reg']
        self.pretrained_model = []
        self.max_seq_len = opt.max_seq_len
            
        # net seq
        self.tcn_channels = list(map(lambda x: int(x.strip()), opt.tcn_channels.split(',')))
        self.net_seq = TCNModel(opt.input_dim, self.tcn_channels, kernel_size=3, dropout=opt.tcn_dropout)

        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.net_reg = FcRegressor(self.tcn_channels[-1], layers, 1, dropout=opt.dropout_rate)
        
        # settings 
        self.target_name = opt.target
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
        self.feature = input['feature'].to(self.device)
        self.mask = input['mask'].to(self.device)
        self.length = input['length']
        if load_label:
            self.target = input[self.target_name].to(self.device)
            
    
    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output = []
        
        for step in range(split_seg_num):
            feature_step = self.feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            prediction = self.forward_step(feature_step)
            
            self.output.append(prediction.squeeze(dim=-1))
            # backward
            if self.isTrain:
                self.optimizer.zero_grad()
                target = self.target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                self.backward_step(prediction, target, mask)
                self.optimizer.step() 
        self.output = torch.cat(self.output, dim=1)
        
        
    def forward_step(self, input):
        hidden = self.net_seq(input)
        prediction, _ = self.net_reg(hidden)
        return prediction
   
   
    def backward_step(self, pred, target, mask):
        """Calculate the loss for back propagation"""
        pred = pred.squeeze() * mask
        target = target * mask
        batch_size = target.size(0)
        self.loss_MSE = self.criterion_reg(pred, target) / batch_size
        self.loss_MSE.backward(retain_graph=False)    
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)
            
            
            
if __name__ == '__main__':
    import sys
    sys.path.append('/data8/hzp/ABAW_VA_2022/code/utils')#
    from tools import calc_total_dim
    from data import create_dataset, create_dataset_with_args

    class test:
        feature_set = 'denseface'
        max_seq_len = 100
        bidirection = False
        input_dim = calc_total_dim(list(map(lambda x: x.strip(), feature_set.split(',')))) #计算出拼接后向量的维度
        regress_layers = '256,128'
        lr = 1e-4
        #weight_decay = 1e-4
        beta1 = 0.5

        batch_size = 8
        epoch_count = 1
        niter=20
        niter_decay=30
        gpu_ids = 0
        isTrain = True
        checkpoints_dir = ''
        name = ''
        cuda_benchmark = ''
        dropout_rate = 0.3
        target = 'arousal'
        loss_type = ''
        dataset_mode = 'seq'
        serial_batches = True
        num_threads = 0
        max_dataset_size = float("inf")
        norm_method = ''
        norm_features = ''
        
        tcn_channels = '512'
        tcn_dropout = 0.2
        kernel_size = 3

    
    opt = test()
    net_a = TcnModel(opt)


    dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options

    total_iters = 0                             # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        for i, data in enumerate(dataset):  # inner loop within one epoch
            total_iters += 1                # opt.batch_size
            epoch_iter += opt.batch_size
            net_a.set_input(data)           # unpack data from dataset and apply preprocessing
            net_a.run()

            print(net_a.output)
            print(net_a.output.shape)