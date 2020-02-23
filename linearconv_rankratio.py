import torch
import torch.nn as nn
import torch.nn.functional as F

# Rank-ratio version
class xCNNlow(torch.nn.Module):
    def __init__(self, channels, filters, kernel_size, padding=1, stride=1, groups=1, rank=1, bias=True):
        super(xCNNlow, self).__init__()
        self.filters = filters
        self.times = 2
        self.kernel_size = kernel_size
        self.channels = channels//groups
        self.padding = padding
        self.stride = stride
        self.biasTrue = bias
        self.rank = rank
        self.groups = groups

        self.conv_weights = nn.Parameter(torch.Tensor(filters//self.times, channels, kernel_size, kernel_size).to(device))
        self.column_weights = nn.Parameter(torch.Tensor(filters-filters//self.times, int((filters//self.times)*self.rank)).to(device))
        self.row_weights = nn.Parameter(torch.Tensor(int((filters//self.times)*self.rank), filters//self.times).to(device))
        
        torch.nn.init.xavier_uniform(self.conv_weights)
        self.column_weights.data.uniform_(-0.1, 0.1)
        self.row_weights.data.uniform_(-0.1, 0.1)
        
        if self.biasTrue:
            self.bias = nn.Parameter(torch.Tensor(filters).to(device))
            self.bias.data.uniform_(-0.1, 0.1)

    def forward(self, input):       
        self.correlated_weights = torch.mm(self.column_weights, torch.mm(self.row_weights,self.conv_weights.reshape(self.filters//self.times,-1)))\
                .reshape(self.filters-self.filters//self.times, self.channels, self.kernel_size, self.kernel_size)       
        if self.biasTrue:
            return F.conv2d(input, torch.cat((self.conv_weights,self.correlated_weights), dim = 0),\
                bias=self.bias, padding=self.padding, stride=self.stride)
        else:
            return F.conv2d(input, torch.cat((self.conv_weights,self.correlated_weights), dim = 0),\
                padding=self.padding, stride=self.stride)