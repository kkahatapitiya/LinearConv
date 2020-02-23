import torch
import torch.nn as nn
import torch.nn.functional as F

# Sparse version
class xCNN(torch.nn.Module):                                                                                                  
    def __init__(self, channels, filters, kernel_size, padding=1, stride=1, groups=1, bias=True):                             
        super(xCNN, self).__init__()                                                                                          
        self.filters = filters                                                                                                
        self.times = 2 #ratio 1/2                                                                                             
        self.kernel_size = kernel_size                                                                                        
        self.channels = channels//groups                                                                                      
        self.padding = padding                                                                                                
        self.stride = stride                                                                                                  
        self.biasTrue = bias                                                                                                  
        self.groups = groups                                                                                                  
                                                                                                                              
        self.counter = 0                                                                                                      
        self.threshold = 0                                                                                                    
        #self.mask = torch.abs(self.linear_weights) > self.threshold                                                          
                                                                                                                              
        self.conv_weights = nn.Parameter(torch.Tensor(filters//self.times, channels, kernel_size, kernel_size).to(device))    
        self.linear_weights = nn.Parameter(torch.Tensor(filters-filters//self.times, filters//self.times).to(device))         
                                                                                                                              
        torch.nn.init.xavier_uniform(self.conv_weights)                                                                       
        self.linear_weights.data.uniform_(-0.1, 0.1)                                                                          
                                                                                                                              
        #self.mask = torch.abs(self.linear_weights) > self.threshold                                                          
                                                                                                                              
        if self.biasTrue:                                                                                                     
            self.bias = nn.Parameter(torch.Tensor(filters).to(device))                                                        
            self.bias.data.uniform_(-0.1, 0.1)                                                                                
                                                                                                                              
        self.mask = nn.Parameter(torch.abs(self.linear_weights) > self.threshold, requires_grad = False)                      
        self.mask.requires_grad = False                                                                                       
                                                                                                                              
    def forward(self, input):                                                                                                 
                                                                                                                              
        self.counter += 1                                                                                                     
        if self.counter == prune_step:                                                                                        
            self.counter = 0                                                                                                  
            self.mask = nn.Parameter(torch.abs(self.linear_weights) > self.threshold, requires_grad = False)                  
            self.percentile = 1. - float(torch.sum(self.mask).item())/(self.mask.shape[0]**2)                                 
            #self.threshold += (req_percentile - self.percentile) * thres_step                                                
            self.threshold += (2./(1.+10**(10*(self.percentile-req_percentile)))-1) * thres_step                              
            print('pruned... %.2f, %.5f' %(self.percentile, self.threshold))                                                  
                                                                                                                              
        self.mask = nn.Parameter(self.mask.type(torch.FloatTensor).to(device), requires_grad = False)                         
        temp = self.linear_weights * self.mask                                                                                
        self.correlated_weights = torch.mm(temp, self.conv_weights.reshape(self.filters//self.times,-1))\
                .reshape(self.filters-self.filters//self.times, self.channels, self.kernel_size, self.kernel_size) 

        if self.biasTrue:                                                                                                     
            return F.conv2d(input, torch.cat((self.conv_weights,self.correlated_weights), dim = 0),\
                bias=self.bias, padding=self.padding, stride=self.stride)                                                     
        else:                                                                                                                 
            return F.conv2d(input, torch.cat((self.conv_weights,self.correlated_weights), dim = 0),\
                padding=self.padding, stride=self.stride) 