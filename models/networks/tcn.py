import torch
import torch.nn as nn
from itertools import repeat
from torch.nn.utils import weight_norm

'''
TCN原文：An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling
在TCN的官方实现的基础上修改：
https://github.com/locuslab/TCN/blob/master/TCN/tcn.py
'''

class Chomp1d(nn.Module):
    '''
    用chomp的原因：因果卷积由于是不能看到未来信息的，所以当前时刻应该是窗口中的最后一个时刻，
    所以padding的时候应该全部padding到输入序列左侧，右侧不加padding。
    但是pytorch的padding是同时在两端pad指定长度。
    因此我们可以先按pytorch的标准做法在两端都pad (k-1)的长度，然后再将最右侧pad的部分去掉。
    input_shape: (bs, embd_dim, seq_len)
    '''
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() # contiguous(): 强制拷贝一份tensor


class SpatialDropout(nn.Module):
    '''
    提出的原文：Efficient Object Localization Using Convolutional Networks. https://arxiv.org/pdf/1411.4280.pdf
    参考实现：https://blog.csdn.net/weixin_43896398/article/details/84762943
    '''
    def __init__(self, dropout):
        super(SpatialDropout, self).__init__()
        self.dropout = dropout
    
    def _make_noise(self, input):
        return input.new().resize_(input.size(0), *repeat(1, input.dim() - 2), input.size(2))
        # torch.new(): 创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致，且无内容。
        # *repeat(1, input.dim() - 2)的作用：在除了第一个和最后一个维度上，其它的维度有几个就在这里对应得创建几个维度，且每个维度的长度为1

    def forward(self, input):
        output = input.clone()
        if not self.training or self.dropout == 0:
            return input
        else:
            noise = self._make_noise(input)
            if self.dropout == 1:
                noise.fill_(0)
            else:
                noise.bernoulli_(1 - self.dropout).div_(1 - self.dropout)
            noise = noise.expand_as(input) #把长度为1的那些维度扩展到与input一致（这样就实现了对一整个通道一起mask）
            output.mul_(noise)
        return output


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, dilation, kernel_size, stride=1, dropout=0.2, use_norm=True, dropout_type='normal'):
        '''
        use_norm: 是否做weight normalization，TCN原文和1st代码中均做了
        dropout_type: ['normal', 'spatial']。'normal'表示用nn.Dropout()，'spatial'表示做spatial dropout。默认就是'normal'，与原代码一致。
        '''
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        if use_norm:
            self.norm_1 = weight_norm # 原文中说用"weight normalization"
            self.conv_1 = self.norm_1(nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
            self.norm_2 = weight_norm # 原文中说用"weight normalization"
            self.conv_2 = self.norm_2(nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation))
        else:
            self.conv_1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
            self.conv_2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp_1 = Chomp1d(padding)
        self.chomp_2 = Chomp1d(padding)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        assert dropout_type in ['spatial', 'normal']
        if dropout_type == 'spatial':
            self.dropout_1 = SpatialDropout(dropout)
            self.dropout_2 = SpatialDropout(dropout)
        else:
            self.dropout_1 = nn.Dropout(dropout)
            self.dropout_2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv_1, self.chomp_1, self.relu_1, self.dropout_1,
                                 self.conv_2, self.chomp_2, self.relu_2, self.dropout_2)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None  #这一行其实就是对应的残差连接那个1×1卷积
        self.relu = nn.ReLU()
        # self.init_weights()

    # def init_weights(self):
    #     self.conv_1.weight.data.normal_(0, 0.01)
    #     self.conv_2.weight.data.normal_(0, 0.01)
    #     if self.downsample is not None:
    #         self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    '''
    TCN (Temporal Convolutional Network) model
    '''
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2, use_norm=True, dropout_type='normal'):
        '''
        num_channels: 一个列表，列表的长度等于这里需要堆叠TemporalBlock的个数，每个元素代表每个块的输出通道的维度，如：(512,)
        '''
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, dilation_size, kernel_size, stride=1, dropout=dropout,
                                     use_norm=True, dropout_type='normal')]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)



if __name__ == '__main__':
    # model = TCNModel(2048, (512,))
    model = TCNModel(2048, [512,])
    # input = torch.rand((8, 60, 2048)) #(bs, seq_len, embd_dim)
    input = torch.rand((8, 2048, 60)) #(bs, embd_dim, seq_len)
    # print(input)
    output = model(input)
    print(model)
    # print(output)
    print(output.shape) # (8, 512, 60)