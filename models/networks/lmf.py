'''
modified by the code:
https://github.com/Justin1904/Low-rank-Multimodal-Fusion
'''

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


class SubNet(nn.Module):
    '''
    The subnetwork that is used in LMF for video and audio in the pre-fusion stage
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


class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, hidden_dims, dropouts, output_dim, rank, use_softmax=False):
        '''
        Args:
            input_dims - a length-2 tuple, contains (audio_dim, video_dim)
            hidden_dims - another length-2 tuple, hidden dims of the sub-networks
            dropouts - a length-3 tuple, contains (audio_dropout, video_dropout, post_fusion_dropout)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            a tensor shaped (2,)
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio, video
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]

        self.audio_hidden = hidden_dims[0]
        self.video_hidden = hidden_dims[1]
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.audio_prob = dropouts[0]
        self.video_prob = dropouts[1]
        self.post_fusion_prob = dropouts[2]

        # define the pre-fusion subnetworks
        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, self.video_prob)

        # define the post_fusion layers
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        # self.post_fusion_layer_1 = nn.Linear((self.text_out + 1) * (self.video_hidden + 1) * (self.audio_hidden + 1), self.post_fusion_dim)
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_hidden + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init teh factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
        '''
        audio_h = self.audio_subnet(audio_x)
        video_h = self.video_subnet(video_x)
        batch_size = audio_h.data.shape[0]

        # next we perform low-rank multimodal fusion
        # here is a more efficient implementation than the one the paper describes
        # basically swapping the order of summation and elementwise product
        if audio_h.is_cuda:
            DTYPE = torch.cuda.FloatTensor
        else:
            DTYPE = torch.FloatTensor

        _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_h), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_h), dim=1)

        fusion_audio = torch.matmul(_audio_h, self.audio_factor) # (rank, bs*seq_len, out_dim)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_zy = fusion_audio * fusion_video # (rank, bs*seq_len, out_dim)
        
        # output = torch.sum(fusion_zy, dim=0).squeeze()
        # use linear transformation instead of simple summation, more flexibility
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output)
        return output


if __name__ == '__main__':
    lmf_hidden = 32
    model = LMF((256, 256), (lmf_hidden, lmf_hidden), (0.3, 0.3, 0.3), output_dim=2, rank=4)
    audio_input = torch.rand((8, 60, 256)) # (bs, seq_len, embd_dim)
    video_input = torch.rand((8, 60, 256)) # (bs, seq_len, embd_dim)
    bs = audio_input.shape[0]
    audio_dim = audio_input.shape[-1]
    video_dim = video_input.shape[-1]
    audio_input = audio_input.reshape(-1, audio_dim) # (bs * seq_len, embd_dim)
    video_input = video_input.reshape(-1, video_dim)
    fusion_tensor = model(audio_input, video_input) # (bs*seq_len, 2)
    fusion_tensor = fusion_tensor.reshape(bs, -1, 2) # (bs, seq_len, 2)
    print(fusion_tensor.shape)
