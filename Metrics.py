import torch
import torch.nn as nn
import numpy as np
import scipy.linalg as linalg

import matplotlib.pyplot as plt
import pandas as pd

#MMD metric
def guassian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):

    n_samples = int(source.size()[0])+int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    
    L2_distance = ((total0-total1)**2).sum(2) 
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
    
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    
    return sum(kernel_val)#/len(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
   
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY -YX)
    return loss

#SSD metric
def SSD(y, y_pred):
    # Assuming y and y_pred are PyTorch tensors and already on the GPU
    return torch.sum((y - y_pred) ** 2, dim=1)  # dim 1 is the signal dimension

#PRD metric
def PRD(y, y_pred):
    # Assuming y and y_pred are PyTorch tensors and already on the GPU
    N = torch.sum((y_pred - y) ** 2, dim=1)
    D = torch.sum((y_pred - torch.mean(y, dim=0)) ** 2, dim=1)
    
    PRD = torch.sqrt(N / D) * 100
    return PRD

#COSS metric
def COSS(y, y_pred):
    # Assuming y and y_pred are PyTorch tensors and already on the GPU
    return torch.nn.functional.cosine_similarity(y,y_pred)