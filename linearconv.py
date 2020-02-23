import torch
import torch.nn as nn
import torch.nn.functional as F

class xCNN(torch.nn.Module):
    def __init__(self, channels, filters, kernel_size, padding=0, stride=1, groups=1, bias=True):
        super(xCNN, self).__init__()
        self.filters = filters
        self.times = 2 #ratio 1/2
        self.kernel_size = kernel_size
        self.channels = channels//groups
        self.padding = padding
        self.stride = stride
        self.biasTrue = bias
        self.groups = groups

        self.conv_weights = nn.Parameter(torch.Tensor(filters//self.times, channels, kernel_size, kernel_size).to(device))
        self.linear_weights = nn.Parameter(torch.Tensor(filters-filters//self.times, filters//self.times).to(device))
        
        torch.nn.init.xavier_uniform(self.conv_weights)
        self.linear_weights.data.uniform_(-0.1, 0.1)
        
        if self.biasTrue:
            self.bias = nn.Parameter(torch.Tensor(filters).to(device))
            self.bias.data.uniform_(-0.1, 0.1)


    def forward(self, input):
        self.correlated_weights = torch.mm(self.linear_weights,self.conv_weights.reshape(self.filters//self.times,-1))\
                .reshape(self.filters-self.filters//self.times, self.channels, self.kernel_size, self.kernel_size)
        
        if self.biasTrue:
            return F.conv2d(input, torch.cat((self.conv_weights,self.correlated_weights), dim = 0),\
                bias=self.bias, padding=self.padding, stride=self.stride)
        else:
            return F.conv2d(input, torch.cat((self.conv_weights,self.correlated_weights), dim = 0),\
                padding=self.padding, stride=self.stride)


#count FLOPs
def count_op_xCNN(m, x, y):
    x = x[0]

    multiply_adds = 1

    cin = m.channels
    cout = m.filters
    kh, kw = m.kernel_size, m.kernel_size
    batch_size = x.size()[0]

    out_h = y.size(2)
    out_w = y.size(3)

    # ops per output element
    # kernel_mul = kh * kw * cin
    # kernel_add = kh * kw * cin - 1
    kernel_ops = multiply_adds * kh * kw
    bias_ops = 1 if m.biasTrue is True else 0
    ops_per_element = kernel_ops + bias_ops

    # total ops
    # num_out_elements = y.numel()
    output_elements = batch_size * out_w * out_h * cout
    conv_ops = output_elements * ops_per_element * cin // 1 #m.groups=1

    # per output element
    total_mul = m.filters//m.times
    total_add = total_mul - 1
    num_elements = (m.filters - m.filters//m.times) * (cin * kh * kw)
    lin_ops = (total_mul + total_add) * num_elements
    total_ops = lin_ops + conv_ops
    print(lin_ops, conv_ops)

    m.total_ops = torch.Tensor([int(total_ops)])

