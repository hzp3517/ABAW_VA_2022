import torch
import numpy as np
import os
import torch.nn.functional as F
# from .base_model import BaseModel
# from .networks.regressor import FcRegressor
# from .networks.transformer import TransformerEncoder#

import sys
sys.path.append('/data8/hzp/ABAW_VA_2022/code')#
from models.base_model import BaseModel#
from models.networks.regressor import FcRegressor#
from models.networks.transformer import TransformerEncoder#

class TransformerSlideModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=256, type=int, help='transformer encoder hidden states')
        parser.add_argument('--num_layers', default=4, type=int, help='number of transformer encoder layers')
        parser.add_argument('--ffn_dim', default=1024, type=int, help='dimension of FFN layer of transformer encoder')
        parser.add_argument('--nhead', default=4, type=int, help='number of heads of transformer encoder')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        parser.add_argument('--target', default='arousal', type=str, help='one of [arousal, valence]')
        parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'l1', 'ccc'], help='use MSEloss or L1loss or CCCloss')
        return parser
    
    
    def __init__(self, opt, logger=None):
        """Initialize the Transformer Slide class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = ['MSE']
        self.model_names = ['_seq', '_reg']
        self.pretrained_model = []
        
        # net seq (already include a linear projection before the transformer encoder)
        if opt.hidden_size == -1:
            opt.hidden_size = min(opt.input_dim // 2, 512)
        self.net_seq = TransformerEncoder(opt.input_dim, opt.num_layers, opt.nhead, \
                                          dim_feedforward=opt.ffn_dim, affine=True, affine_dim=opt.hidden_size)
        
        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        self.hidden_size = opt.hidden_size
        self.net_reg = FcRegressor(opt.hidden_size, layers, 1, dropout=opt.dropout_rate)
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
        if load_label:
            self.mask = input['mask'].to(self.device)
            self.target = input[self.target_name].to(self.device)

    def run(self):
        """After feed a batch of samples, Run the model."""
        prediction = self.forward_step(self.feature)
        # backward
        if self.isTrain:
            self.optimizer.zero_grad()
            self.backward_step(prediction, self.target, self.mask)
            self.optimizer.step() 
        self.output = prediction.squeeze(-1)
        
    def forward_step(self, input):
        out, hidden_states = self.net_seq(input) # hidden_states: layers * (seq_len, bs, ft_dim)
        last_hidden = hidden_states[-1].transpose(0, 1) # (bs, seq_len, ft_dim)
        prediction, _ = self.net_reg(last_hidden)
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
        feature_set = 'vggish'
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
        loss_type = 'mse'
        dataset_mode = 'seq_slide'
        serial_batches = True
        num_threads = 0
        max_dataset_size = float("inf")
        norm_method = ''
        norm_features = ''
        win_len = 300
        hop_len = 50
        
        hidden_size = 256
        num_layers = 4
        ffn_dim = 1024
        nhead = 4

    
    opt = test()
    net_a = TransformerSlideModel(opt)


    dataset, val_dataset = create_dataset_with_args(opt, set_name=['train', 'val'])  # create a dataset given opt.dataset_mode and other options

    total_iters = 0                             # the total number of training iterations
    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        
        # # # --train----
        # for i, data in enumerate(dataset):  # inner loop within one epoch
        #     total_iters += 1                # opt.batch_size
        #     epoch_iter += opt.batch_size
        #     net_a.set_input(data)           # unpack data from dataset and apply preprocessing
        #     net_a.run()

        #     print(net_a.output)
        #     print(net_a.output.shape)
        
        
        # --val---
        for i, data in enumerate(val_dataset):  # inner loop within one epoch
            net_a.set_input(data)           # unpack data from dataset and apply preprocessing
            net_a.isTrain = False
            with torch.no_grad():
                net_a.run()

            print(net_a.output)
            print(net_a.output.detach().cpu().squeeze(0).numpy())
            print(net_a.output.shape)