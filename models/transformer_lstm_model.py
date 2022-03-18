'''
在transformer encoder上再加一层lstm-xl，若使用预训练过的transformer encoder，则直接将该层以及前面的fc层冻结
'''

import torch
import numpy as np
import os
import torch.nn.functional as F
# from .base_model import BaseModel
# from .networks.regressor import FcRegressor
# from .networks.transformer import TransformerEncoder#


import sys
sys.path.append('/data2/hzp/ABAW_VA_2022/code')
from models.base_model import BaseModel
from models.networks.regressor import FcRegressor
# from models.networks.fft import FFTEncoder
from models.networks.fc_encoder import FcEncoder
from models.networks.transformer import TransformerEncoder
from models.networks.lstm_encoder import LSTMEncoder
from models.loss import CCCLoss, MSELoss, MultipleLoss
from utils.bins import get_center_and_bounds


class TransformerLstmModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.add_argument('--max_seq_len', type=int, default=100, help='max sequence length of transformer')
        parser.add_argument('--regress_layers', type=str, default='256,128', help='256,128 for 2 layers with 256, 128 nodes respectively')
        parser.add_argument('--hidden_size', default=256, type=int, help='transformer encoder hidden states')
        parser.add_argument('--num_layers', default=4, type=int, help='number of transformer encoder layers')
        parser.add_argument('--ffn_dim', default=1024, type=int, help='dimension of FFN layer of transformer encoder')
        parser.add_argument('--nhead', default=4, type=int, help='number of heads of transformer encoder')
        parser.add_argument('--dropout_rate', default=0.3, type=float, help='drop out rate of FC layers')
        parser.add_argument('--target', default='arousal', type=str, choices=['valence', 'arousal', 'both'], help='one of [arousal, valence]')
        parser.add_argument('--use_pe', action='store_true', help='whether to use position encoding')
        parser.add_argument('--encoder_type', type=str, default='transformer', choices=['transformer'], help='whether to use position encoding')
        parser.add_argument('--loss_type', type=str, default='mse', nargs='+',
                            choices=['mse', 'ccc', 'batch_ccc', 'amse', 'vmse', 'accc', 'vccc', 'batch_accc',
                                     'batch_vccc', 'ce'])
        parser.add_argument('--loss_weights', type=float, default=1, nargs='+')
        parser.add_argument('--cls_loss', default=False, action='store_true', help='whether to cls and average as loss')
        parser.add_argument('--cls_weighted', default=False, action='store_true', help='whether to use weighted cls')

        parser.add_argument('--transformer_pretrained', default=False, action='store_true', help='whether to use the pretrained transformer')
        parser.add_argument('--pth_path', type=str, default='', help='the pth file path of the pretrained transformer model')
        parser.add_argument('--residual', type=str, default='y', choices=['y', 'n'], help='whether to use the residual connection to skip the transformer encoder')
        return parser

    def __init__(self, opt, logger=None):
        """Initialize the LSTM autoencoder class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        super().__init__(opt, logger)
        self.loss_names = ['reg']
        self.model_names = ['_fc', '_te', '_lstm', '_reg']
        
        self.max_seq_len = opt.max_seq_len
        self.use_pe = opt.use_pe
        self.encoder_type = opt.encoder_type
        self.loss_type = opt.loss_type
        if opt.hidden_size == -1:
            opt.hidden_size = min(opt.input_dim // 2, 512)
        self.hidden_size = opt.hidden_size
        self.cls_loss = opt.cls_loss
        if 'ce' in opt.loss_type:
            opt.cls_loss = True
        self.target_name = opt.target
        self.transformer_pretrained = opt.transformer_pretrained
        self.residual = opt.residual

        if self.cls_loss:
            bin_centers, bin_bounds = get_center_and_bounds(opt.cls_weighted)
            self.bin_centers = dict([(key, np.array(value)) for key, value in bin_centers.items()])
            self.bin_bounds = dict([(key, np.array(value)) for key, value in bin_bounds.items()])

        # net fc fusion
        self.net_fc = FcEncoder(opt.input_dim, [opt.hidden_size], dropout=0.1, dropout_input=False)

        # net transformer encoder (already include a linear projection before the transformer encoder)
        if self.encoder_type == 'transformer':
            self.net_te = TransformerEncoder(opt.hidden_size, opt.num_layers, opt.nhead, \
                                            dim_feedforward=opt.ffn_dim, affine=False, use_pe=self.use_pe)

        # load pretrained weights
        if self.transformer_pretrained:
            self.pretrained_model = ['_fc', '_te']
            # 加载预训练模型权重
            pre_model_path = opt.pth_path.strip()
            pre_model = torch.load(pre_model_path)
            fc_model_dict = self.net_fc.state_dict()
            fc_state_dict = {k.replace('affine', 'module.0'): v for k, v in pre_model.items() if k.startswith('affine')}
            fc_model_dict.update(fc_state_dict)
            self.net_fc.load_state_dict(fc_model_dict)
            te_model_dict = self.net_te.state_dict()
            te_state_dict = {k: v for k, v in pre_model.items() if k in te_model_dict.keys()}
            te_model_dict.update(te_state_dict)
            self.net_te.load_state_dict(te_model_dict)
        else:
            self.pretrained_model = []

        # net lstm-xl
        self.net_lstm = LSTMEncoder(opt.hidden_size, opt.hidden_size)

        # net regression
        layers = list(map(lambda x: int(x), opt.regress_layers.split(',')))
        output_dim = 22 if self.cls_loss else 1
        if self.target_name == 'both':
            output_dim *= 2
        self.net_reg = FcRegressor(opt.hidden_size, layers, output_dim, dropout=opt.dropout_rate)

        # settings
        if self.isTrain:
            if self.isTrain:
                self.criterion_reg = MultipleLoss(reduction='mean', loss_names=opt.loss_type, weights=opt.loss_weights)
            
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
            if self.target_name == 'both':
                self.target = torch.stack([input['valence'], input['arousal']], dim=2).to(self.device)
            else:
                self.target = input[self.target_name].to(self.device)

            if self.cls_loss:
                if self.target_name == 'both':
                    #[B, L, 2]
                    self.cls_target = torch.stack([input['valence_cls'], input['arousal_cls']], dim=2).to(self.device)
                else:
                    self.cls_target = input[self.target_name].to(self.device)


    def run(self):
        """After feed a batch of samples, Run the model."""
        batch_size = self.target.size(0)
        batch_max_length = torch.max(self.length).item()
        # calc num of splited segments
        split_seg_num = batch_max_length // self.max_seq_len + int(batch_max_length % self.max_seq_len != 0)
        # forward in each small steps
        self.output = []
        previous_h = torch.zeros(1, batch_size, self.hidden_size).float().to(self.device) 
        previous_c = torch.zeros(1, batch_size, self.hidden_size).float().to(self.device)
        for step in range(split_seg_num):
            feature_step = self.feature[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
            prediction, (previous_h, previous_c), logits = self.forward_step(feature_step, (previous_h, previous_c))
            previous_h = previous_h.detach()
            previous_c = previous_c.detach()
            self.output.append(prediction.squeeze(dim=-1))
            # backward
            if self.isTrain:
                self.optimizer.zero_grad()  
                target = self.target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                mask = self.mask[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                cls_target = None if not self.cls_loss else self.cls_target[:, step*self.max_seq_len: (step+1)*self.max_seq_len]
                self.backward_step(prediction, target, mask, logits, cls_target)
                self.optimizer.step() 
        self.output = torch.cat(self.output, dim=1)
    

    def forward_step(self, input, states):
        if self.transformer_pretrained: # if use the pretrained transformer encoder, freeze that model.
            with torch.no_grad():
                fusion = self.net_fc(input)
                out, te_hidden_states = self.net_te(fusion) # te_hidden_states: layers * (seq_len, bs, hidden_size)
        else:
            fusion = self.net_fc(input)
            out, te_hidden_states = self.net_te(fusion) # te_hidden_states: layers * (seq_len, bs, hidden_size)
        last_hidden = te_hidden_states[-1].transpose(0, 1) # (bs, seq_len, hidden_size)
        if self.residual == 'y': # residual connection
            last_hidden = fusion + last_hidden
            # print('res:', last_hidden)
        lstm_hidden, (h, c) = self.net_lstm(last_hidden, states)
        prediction, _ = self.net_reg(lstm_hidden)
        logits = None
        if self.cls_loss:
            logits = prediction.reshape(prediction.shape[:-1] + (22, 2)).transpose(1, 2) #[B, L, 44] -> [B, L, 22, 2] -> [B, 22, L, 2]
            if self.target_name == 'both':
                prediction = prediction.reshape(prediction.shape[:-1] + (22, 2))
                weights = torch.cat([torch.FloatTensor(self.bin_centers['valence']),
                                     torch.FloatTensor(self.bin_centers['arousal'])]).reshape(1, 1, -1, 2).cuda()
            else:
                weights = torch.FloatTensor(self.bin_centers[self.target_name]).reshape(1, 1, -1, 1).cuda()
                prediction = prediction.unsqueeze(-1)
            prediction = F.softmax(prediction, dim=-2)
            prediction = torch.sum(prediction * weights, dim=-2)
        return prediction, (h, c), logits
    
    
    def backward_step(self, pred, target, mask, logits=None, cls_target=None):
        """Calculate the loss for back propagation"""
        mask = mask.unsqueeze(-1)  # -> [B, L, 1]
        if self.target_name != 'both':
            target = target.unsqueeze(-1)  # -> [B, L, 1]
        else:
            mask = mask.expand(mask.shape[0], mask.shape[1], 2)  # -> [B, L, 2]

        self.loss_reg = self.criterion_reg(pred, target, mask, logits, cls_target)
        self.loss_reg.backward(retain_graph=False)
        for model in self.model_names:
            torch.nn.utils.clip_grad_norm_(getattr(self, 'net'+model).parameters(), 5)


if __name__ == '__main__':
    import sys
    sys.path.append('/data2/hzp/ABAW_VA_2022/code/utils')#
    from tools import calc_total_dim
    from data import create_dataset, create_dataset_with_args

    class test:
        feature_set = 'affectnet,vggish,wav2vec'
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
        niter_warmup=4
        niter_total=70
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
        
        hidden_size = 256
        num_layers = 4
        ffn_dim = 1024
        nhead = 4
        use_pe = True
        encoder_type = 'transformer'
        cls_loss = False
        cls_weighted = False
        loss_weights = 1
        transformer_pretrained = True
        pth_path = '/data2/hzp/ABAW_VA_2022/code/checkpoints/3-16/transformer_both_affectnet-vggish-wav2vec_bs16_lr2e-05_dp0.3_seq250_reg-256-256_hidden256_layers4_ffn1024_nhead4_batch_ccc/13_net_seq.pth'
        residual = True

    
    opt = test()
    net_a = TransformerLstmModel(opt)


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