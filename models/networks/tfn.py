'''
modified by the code:
https://github.com/Justin1904/TensorFusionNetworks/blob/master/model.py
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_uniform, xavier_normal, orthogonal

class SubNet(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(SubNet, self).__init__()
        self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        normed = self.norm(x)
        dropped = self.drop(normed)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3


class TFN(nn.Module):
    '''
    Implements the Tensor Fusion Networks for multimodal sentiment analysis as is described in:
    Zadeh, Amir, et al. "Tensor fusion network for multimodal sentiment analysis." EMNLP 2017 Oral.
    '''

    def __init__(self, input_dims, hidden_dims, dropouts):
        '''
        Args:
            input_dims - a length-2 tuple, contains (audio_dim, video_dim)
            hidden_dims - another length-2 tuple, similar to input_dims
            dropouts - a length-3 tuple, contains (audio_dropout, video_dropout, post_fusion_dropout)
        Output:
            a fusion tensor
        '''
        super(TFN, self).__init__()

        # dimensions are specified in the order of audio, video and text
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.post_fusion_prob = dropouts[2]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)

    def forward(self, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
        '''
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        batch_size = audio_h.data.shape[0]

        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)

        # _audio_h has shape (batch_size, audio_in + 1), _video_h has shape (batch_size, _video_in + 1)
        # we want to perform outer product between the two batch, hence we unsqueenze them to get
        # (batch_size, audio_in + 1, 1) X (batch_size, 1, video_in + 1)
        # fusion_tensor will have shape (batch_size, audio_in + 1, video_in + 1)
        fusion_tensor = torch.bmm(_audio_h.unsqueeze(2), _video_h.unsqueeze(1))

        # in the end we don't keep the 2-D tensor, instead we flatten it
        fusion_tensor = fusion_tensor.view(batch_size, -1)

        post_fusion_dropped = self.post_fusion_dropout(fusion_tensor)

        return post_fusion_dropped


if __name__ == '__main__':
    tfn_hidden = 32
    model = TFN((470, 342), (tfn_hidden, tfn_hidden), (0.3, 0.3, 0.3))
    audio_input = torch.rand((8, 60, 470)) # (bs, seq_len, embd_dim)
    video_input = torch.rand((8, 60, 342)) # (bs, seq_len, embd_dim)
    bs = audio_input.shape[0]
    audio_dim = audio_input.shape[-1]
    video_dim = video_input.shape[-1]
    audio_input = audio_input.reshape(-1, audio_dim) # (bs * seq_len, embd_dim)
    video_input = video_input.reshape(-1, video_dim)
    fusion_tensor = model(audio_input, video_input) # (bs*seq_len, (audio_dim+1)*(video_dim+1))
    fusion_dim = (tfn_hidden + 1) * (tfn_hidden + 1)
    fusion_tensor = fusion_tensor.reshape(bs, -1, fusion_dim)
    print(fusion_tensor.shape)
