import torch

def corrLoss(feat):
    loss = 0
    param = {}
    for i in feat.named_parameters():
        if 'conv_weights' in i[0]:
            dat = i[1]
            corr = corrcoef(dat.reshape(dat.shape[0], -1))
            loss += torch.sum(torch.abs(corr - torch.eye(corr.shape[0]).to(device)))
    return loss