'''
利用在vggface2上预训练过的resnet-50模型，在本数据集上做VA回归任务
'''
import torch
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# from .base_model import BaseModel
# from .networks.resnet import ResNet, Bottleneck, load_resnet50_weight
# from .networks.classifier import FcRegressor

import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code/')#
from models.base_model import BaseModel#
from models.networks.resnet import ResNet, Bottleneck, load_resnet50_weight#
from models.networks.regressor import FcRegressor#


class ResnetModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--arch_type', type=str, default='resnet50_ft', choices=['none', 'resnet50_ft', 'resnet50_scratch'], 
                            help='resnet50_ft:      ResNet-50 which are first pre-trained on MS1M, and then fine-tuned on VGGFace2; \
                                  resnet50_scratch: ResNet-50 trained from scratch on VGGFace2; \
                                  none:             ResNet-50 without pretrained')
        parser.add_argument('--regress_layers', default='256', type=str,
                            help='size of classifier hidden size, split by comma')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate')
        parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1'], help='use MSEloss or L1loss')
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the BaselineModel class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        N_IDENTITY = 8631  # the number of identities in VGGFace2 for which ResNet and SENet are trained. Pretrained weights fc layer has 8631 outputs.
        self.loss_names = [opt.loss_type] # get_current_losses: getattr(self, 'loss_' + name)
        self.pretrain = False if opt.arch_type=='none' else True
        self.arch_type = opt.arch_type
        self.regress_layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.dropout_rate = opt.dropout_rate
        self.model_names = ['_resnet', '_reg']

        self.net_resnet = ResNet(Bottleneck, [3, 4, 6, 3], include_top=False)
        self.net_reg = FcRegressor(2048, self.regress_layers, 2, dropout=self.dropout_rate)

        # 加载预训练模型权重
        weight_file = '/data2/hzp/pretrained_models/resnet50_ft_weight.pkl'
        if self.pretrain:
            self.pretrained_model = ['_resnet']
            self.net_resnet = load_resnet50_weight(weight_file, num_classes=N_IDENTITY, include_top=False)
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
            self.target = input['label'].to(self.device)
    
    def run(self):
        """After feed a batch of samples, Run the model."""
        # forward
        prediction = self.forward_step()
        
        # backward
        if self.isTrain:
            self.optimizer.zero_grad()
            self.backward_step(prediction, self.target)
            self.optimizer.step()
        
        self.output = prediction
    
    def forward_step(self):
        ft = self.net_resnet(self.images) # (bs, dim=2048)
        pred, _ = self.net_reg(ft) # (bs, 2)
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
        gpu_ids = [7]
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
        
        img_type = 'color'
        read_length = 20000
        norm_type = 'reduce_mean'

        arch_type = 'resnet50_ft'
        regress_layers = '256'
        dropout_rate = 0.3
        loss_type = 'mse'

    opt = test()
    net_a = ResnetModel(opt)
    
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