'''
利用在fer+上预训练过的densenet模型，在本数据集上做VA回归任务
'''
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .base_model import BaseModel
# from .networks.dense_net import DenseNetEncoder
# from .networks.classifier import FcRegressor

import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code/')#
from models.base_model import BaseModel#
from models.networks.dense_net import DenseNetEncoder#
from models.networks.regressor import FcRegressor#


class DensefaceModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--pretrain', type=str, default='y', choices=['y', 'n'],
                            help='whether if the densenet model is pretrained on the FER+ dataset')
        parser.add_argument('--frozen_dense_blocks', type=int, default=0,
                            help='how many dense blocks need to freeze during fine-tuning (0~3)')
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
        self.pretrain = True if opt.pretrain=='y' else False
        self.frozen_dense_blocks = opt.frozen_dense_blocks
        self.regress_layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.dropout_rate = opt.dropout_rate
        self.task = opt.task
        self.model_names = ['_dense', '_reg']
        
        self.net_dense = DenseNetEncoder(frozen_dense_blocks=self.frozen_dense_blocks)
        if self.task == 'v' or self.task == 'a':
            self.net_reg = FcRegressor(342, self.regress_layers, 1, dropout=self.dropout_rate)
        else:
            self.net_reg = FcRegressor(342, self.regress_layers, 2, dropout=self.dropout_rate)
        
        if self.pretrain:
            self.pretrained_model = ['_dense']
            # 加载预训练模型权重
            # pre_model_path = "/data7/MEmoBert/emobert/exp/face_model/densenet100_adam0.001_0.0/ckpts/model_step_43.pt"
            pre_model_path = "/data2/hzp/pretrained_models/denseface_model_step_43.pt"
            pre_model = torch.load(pre_model_path)
            new_model_dict = self.net_dense.state_dict()
            state_dict = {k: v for k, v in pre_model.items() if k in new_model_dict.keys()}
            new_model_dict.update(state_dict)
            self.net_dense.load_state_dict(new_model_dict)
        else:
            self.pretrained_model = []
        
        #settings
        if self.isTrain:
            if opt.loss_type == 'mse':
                self.criterion_reg = torch.nn.MSELoss()
            else:
                self.criterion_reg = torch.nn.L1Loss()
            
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            paremeters = [{'params': getattr(self, 'net'+net).parameters()} for net in self.model_names]
            self.optimizer = torch.optim.Adam(paremeters, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)
        
    def set_input(self, input, load_label=True):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        """
        self.images = input['img'].to(self.device) # (bs, channel, h, w)
        if load_label:
            if self.task == 'v':
                self.target = input['label'].transpose(1, 0)[0].to(self.device)
            elif self.task == 'a':
                self.target = input['label'].transpose(1, 0)[1].to(self.device)
            else:
                self.target = input['label'].to(self.device)
    
    def run(self):
        """After feed a batch of samples, Run the model."""
        # forward
        prediction = self.forward_step()
        if self.task == 'v' or self.task == 'a':
            prediction = prediction.squeeze() # (bs,)
        
        # backward
        if self.isTrain:
            self.optimizer.zero_grad()
            self.backward_step(prediction, self.target)
            self.optimizer.step()
        
        self.output = prediction
    
    def forward_step(self):
        ft = self.net_dense(self.images) # (bs, dim=342)
        pred, _ = self.net_reg(ft) # (bs, 2) or (bs, 1)
        return pred
    
    def backward_step(self, pred, target):
        self.loss_mse = self.criterion_reg(pred, target)
        self.loss_mse.backward(retain_graph=False)


if __name__ == '__main__':
    from data import create_dataset, create_dataset_with_args

    class test:
        feature_set = 'None'
        input_dim = None
        lr = 1e-4
        beta1 = 0.5
        # batch_size = 3
        batch_size = 64
        epoch_count = 1
        niter = 20
        niter_decay=30
        gpu_ids = [1]
        isTrain = True
        checkpoints_dir = ''
        name = ''
        cuda_benchmark = ''
        dataset_mode = 'frame_toy'
        num_threads = 0
        max_dataset_size = float("inf")
        serial_batches = True
        lr_policy = ''
        init_type = 'kaiming'
        init_gain = 0.02
        verbose = ''
        
        img_type = 'gray'
        read_length = 20000
        norm_type = 'norm'

        # pretrain = 'n'
        pretrain = 'y'
        frozen_dense_blocks = 0
        regress_layers = '256'
        dropout_rate = 0.3
        loss_type = 'mse'
        # task = 'v+a'
        task = 'v'

    opt = test()
    net_a = DensefaceModel(opt)
    
    dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options

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