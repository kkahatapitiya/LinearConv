import torch
import torchvision
import torchvision.transforms as transforms
#import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
from torchsummary import summary
import os
import math
import summ
#from thop import profile

import argparse
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--type", type=str, required=True, help="rank ratio or sparse")
ap.add_argument("-r", "--rank", type=float, default=1, help="rank of Matrix")
ap.add_argument("-s", "--sparsity", type=float, default=0, help="sparsity of Matrix")
args = vars(ap.parse_args())
rank = args['rank'] #0.75
sparsity = args['sparsity'] #0.25
exp_type = args['type']
if exp_type == 'rank':
    netType = 'BASE1xl'
elif exp_type == 'sparse':
    netType = 'BASE1x'
reg_const_2 = 0.01
req_percentile = sparsity
thres_step = 0.00001 #0.001
prune_step = 500
#classi_new

epochs = 250
start_epoch = 0
batch_size = 50 #49000, 1000
directory = './'
checkpoint_dir = directory+'ckpt/base_xcnn/'
root = directory+'dataset/'
load_ckpt = False
load_ckpt_num = 0
load_path = checkpoint_dir+'net_epoch_'+str(load_ckpt_num)+'.ckpt'
time_every = 100
verbose = True
loss_every = 980

#rank = 1
reg_const = 0.01
reg_per_batches = 10
lrate = 0.001
# NO MILESTONES USED ////////////////////

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root=root, train=True,download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root=root, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
dataiter = iter(trainloader)
images, labels = dataiter.next()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

########################################################################################################################
########################################################################################################################

#***********************************
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

def l1Loss(feat):                                                                                                             
    loss = 0                                                                                                                  
    param = {}                                                                                                                
    for i in feat.named_parameters():                                                                                         
        if 'linear_weights' in i[0]:                                                                                          
            dat = i[1]                                                                                                        
            #corr = corrcoef(dat.reshape(dat.shape[0], -1))                                                                   
            loss += torch.sum(torch.abs(dat))                                                                                 
    return loss 

def summary(model, input):
    with summ.TorchSummarizeDf(model) as tdf:
        x = torch.rand(input).to(device)
        y = model(x)
        df = tdf.make_df()
    print(df)

def count_op_xCNNlow(m, x, y):
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
    conv_ops = output_elements * ops_per_element * cin // m.groups

    # per output element
    total_mul_1 = m.filters//m.times
    total_add_1 = total_mul_1 - 1
    num_elements_1 = m.rank * (cin * kh * kw) # (m.filters - m.filters//m.times)
    total_mul_2 = m.rank
    total_add_2 = total_mul_2 - 1
    num_elements_2 = (m.filters - m.filters//m.times) * (cin * kh * kw) # (m.filters - m.filters//m.times)
    lin_ops = (total_mul_1 + total_add_1) * num_elements_1 + (total_mul_2 + total_add_2) * num_elements_2
    total_ops = lin_ops + conv_ops
    print(lin_ops, conv_ops)

    m.total_ops = torch.Tensor([int(total_ops)])

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

def corrcoef(x):
    mean_x = torch.mean(x, dim=1, keepdim=True)
    xm = x.sub(mean_x.expand_as(x))
    c = xm.mm(xm.t())
    c = c / (x.size(1) - 1)
    d = torch.diag(c)
    stddev = torch.pow(d, 0.5)
    c = c/stddev[:,None]
    c = c/stddev[None,:]
    c = torch.clamp(c, -1.0, 1.0)
    return c

def corrLoss(feat):
    loss = 0
    param = {}
    for i in feat.named_parameters():
        if 'conv_weights' in i[0]:
            dat = i[1]
            corr = corrcoef(dat.reshape(dat.shape[0], -1))
            loss += torch.sum(torch.abs(corr - torch.eye(corr.shape[0]).to(device)))
    return loss

########################################################################################################################
########################################################################################################################

cfg = {
    'VGG11n': [(64,3,'N'), 'M', (128,3,'N'), 'M', (256,3,'N'), (256,3,'N'), 'M', (512,3,'N'), (512,3,'N'), 'M', (512,3,'N'), (512,3,'N'), 'M'],
    'VGG11x': [(64,3,'X'), 'M', (128,3,'X'), 'M', (256,3,'X'), (256,3,'X'), 'M', (512,3,'X'), (512,3,'X'), 'M', (512,3,'X'), (512,3,'X'), 'M'],
    'VGG11xl': [(64,3,'XL'), 'M', (128,3,'XL'), 'M', (256,3,'XL'), (256,3,'XL'), 'M', (512,3,'XL'), (512,3,'XL'), 'M', (512,3,'XL'), (512,3,'XL'), 'M'],
    'BASE1n': [(32,3,'N'), 'M', (64,3,'N'), 'M', (128,3,'N'), 'M', (256,3,'N'), 'M'],
    'BASE1x': [(32,3,'X'), 'M', (64,3,'X'), 'M', (128,3,'X'), 'M', (256,3,'X'), 'M'],
    'BASE1xl': [(32,3,'XL'), 'M', (64,3,'XL'), 'M', (128,3,'XL'), 'M', (256,3,'XL'), 'M']
}

class NET(nn.Module):
    def __init__(self, version):
        super(NET, self).__init__()
        self.in_channels = 3
        self.index = 0
        self.feat = []
        while self.index < len(cfg[version]):
            self.feat.append(self._make_layers(cfg[version]))
        self.feat = nn.Sequential(*self.feat)
        self.fc = nn.Linear(1024, 10)
    def forward(self, x):
        out = x
        for layer in self.feat:
            out = layer(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    def _make_layers(self, cfg):
        layers = []
        for x in cfg[self.index:]:
            if x == 'M':
                self.index += 1
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                return nn.Sequential(*layers)
            else:
                self.index += 1
                F, K, Ltype = x
                if Ltype == 'X':
                    layers += [xCNN(self.in_channels, F, K),
                               nn.BatchNorm2d(F),
                               nn.ReLU(inplace=True)]
                elif Ltype == 'XL':
                    layers += [xCNNlow(self.in_channels, F, K, rank=rank),
                               nn.BatchNorm2d(F),
                               nn.ReLU(inplace=True)]
                elif Ltype == 'N':
                    layers += [nn.Conv2d(self.in_channels, F, kernel_size=K, padding=K//2),
                               nn.BatchNorm2d(F),
                               nn.ReLU(inplace=True)]
                self.in_channels = F

                                                            
net = NET(netType).to(device)
summary(net, (1,3,32,32))
#***********************************
#flops, params = profile(net, input_size=(1, 3, 32,32), custom_ops={xCNN: count_op_xCNN, xCNNlow: count_op_xCNNlow})
#print(flops,params)

criterion = nn.CrossEntropyLoss()

opt = optim.Adam(net.parameters(), lr=lrate)

if os.path.isfile(load_path) and load_ckpt:
    checkpoint = torch.load(load_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    opt.load_state_dict(checkpoint['opt'])
    start_epoch = checkpoint['epoch'] + 1
    print('model loaded')


for epoch in range(start_epoch, epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    start = time.time()
    for ind, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = net(inputs)

        opt.zero_grad()
        loss = criterion(outputs, labels)
        total_loss = loss

        #***********************************
        reg_loss = 0; l1loss = 0;
        if ind % reg_per_batches == reg_per_batches - 1:
            for layer in net.feat:
                reg_loss += corrLoss(layer)
                if exp_type == 'sparse':
                    l1loss += l1Loss(layer) 
            total_loss = loss + reg_const * reg_loss + reg_const_2 * l1loss  
        total_loss.backward()
        opt.step()

        running_loss += loss.item()

        if ind % time_every == time_every - 1 and verbose:
            end = time.time()
            print('[%d, %5d, time: %d ms loss:%.3f reg:%.3f total:%.3f]' %(epoch + 1, ind + 1, (end-start)*1000,
            	loss, reg_loss, total_loss))
            start = time.time()

        if ind % loss_every == loss_every - 1:
            test_loss = 0
            accuracy = 0
            net.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = net.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()                  
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            end = time.time()
            print('time: %d ms, Epoch: %d/%d, Train loss: %.3f, Test loss: %.3f, Test accuracy: %.3f'
                %((end-start)*1000, epoch+1, epochs, running_loss/len(trainloader), test_loss/len(testloader), accuracy/len(testloader)))
            running_loss = 0.
            start = time.time()
            net.train()

    if epoch % 20 == 0:
        save_dict = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'opt': opt.state_dict(),
            }
        torch.save(save_dict, checkpoint_dir+'net_epoch_'+str(epoch)+'.ckpt')

print('Finished Training')

