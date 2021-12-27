#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:51:49 2021
@author: eddie
"""
import numpy as np
import torch
import torch.nn as nn

class LDW_down(nn.Module):
    def __init__(self, kernel_size = 2, stride = None, learnable = True):
        super(LDW_down, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 2
        self.learnable = learnable
        
        if kernel_size == 5:
            self.low_pass_filter = torch.nn.Parameter(torch.tensor([[[[-0.0761025,  0.3535534, 0.8593118, 0.3535534, -0.0761025]]]]), requires_grad = learnable)
            self.high_pass_filter = torch.nn.Parameter(torch.tensor([[[[-0.0761025, -0.3535534, 0.8593118, -0.3535534, -0.0761025]]]]), requires_grad = learnable)
        elif kernel_size == 7:
            self.low_pass_filter = torch.nn.Parameter(torch.tensor([[[[-0.0076129,  -0.073710695, 0.3622055, 0.8524323, 0.3622055, -0.073710695, -0.0076129]]]]), requires_grad = learnable)
            self.high_pass_filter = torch.nn.Parameter(torch.tensor([[[[0.0076129,  -0.073710695, -0.3622055, 0.8524323, -0.3622055, -0.073710695, 0.0076129]]]]), requires_grad = learnable)
        elif kernel_size == 9:
            self.low_pass_filter = torch.nn.Parameter(torch.tensor([[[[0.02807382, -0.060944743, -0.073386624, 0.41472545, 0.7973934, 0.41472545, -0.073386624, -0.060944743, 0.02807382]]]]), requires_grad = learnable)
            self.high_pass_filter = torch.nn.Parameter(torch.tensor([[[[0.02807382, 0.060944743, -0.073386624, -0.41472545, 0.7973934, -0.41472545, -0.073386624, 0.060944743, 0.02807382]]]]), requires_grad = learnable)
        
    def __repr__(self):
        struct = "Lifting_down(kernel_size={}, stride={}, learnable{})".format(self.kernel_size, self.stride, self.learnable)
        return struct
    
    def regular_term_loss(self):
        # low pass filter square sum = 1, low pass filter sum = sqrt(2)
        low_square_sum = (torch.pow(torch.sum(torch.pow(self.low_pass_filter, 2), dim = 3) - 1, 2)).squeeze(-1)
        low_sum = torch.pow(torch.sum(self.low_pass_filter) - 2**(1/2), 2).squeeze(-1)
        constraint1 = low_square_sum + low_sum
        
        # high pass filter sum = 0 & high pass filter square sum = 1 => limit high pass to unit length
        high_square_sum = torch.pow(1 - torch.sum(torch.pow(self.high_pass_filter, 2), dim = 3), 2)
        constraint2 = (high_square_sum + torch.pow(torch.sum(self.high_pass_filter, dim = 3), 2)).squeeze(-1)
        
        # constraint3 => H'H = 1, L'L = 1
        constraint3 = torch.pow(2 - (torch.sum(torch.pow(self.low_pass_filter, 2), dim = 3) + torch.sum(torch.pow(self.high_pass_filter, 2), dim = 3)), 2).squeeze(-1)
        
        # constraint4 => symmetry
        low_symmetry = 0
        high_symmetry = 0
        for i in range(self.kernel_size // 2):
            low_symmetry += torch.pow(self.low_pass_filter[:, :, :, i] - self.low_pass_filter[:, :, :, -(i + 1)], 2)
            high_symmetry += torch.pow(self.high_pass_filter[:, :, :, i] - self.high_pass_filter[:, :, :, -(i + 1)], 2)
        constraint4 = low_symmetry + high_symmetry

        return torch.mean(constraint1 + constraint2 + constraint3 + constraint4).squeeze(-1).squeeze(-1)
    
    def switch_data(self, x, y, dim):
        if x.get_device() == -1:
            device = "cpu"
        else:
            device = x.get_device()
        combine = torch.cat([x, y], dim = dim)
        idx_old = torch.arange(0, x.shape[dim]).to(device)
        idx_old = idx_old.repeat_interleave(2)
        idx_old[1::2] += x.shape[dim]
        if dim == 2:
            combine = combine[:, :, idx_old, :]
        elif dim == 3:
            combine = combine[:, :, :, idx_old]
        return combine
    
    def up(self, x):
        if x.get_device() == -1:
            device = "cpu"
        else:
            device = x.get_device()
            
        # pad the feature map
        batch, channel, height, width = x.shape
        
        # calculate the balance weight
        h_sum = torch.sum(self.high_pass_filter)
        l_sum = torch.sum(self.low_pass_filter)
        weight1 = torch.sum(self.high_pass_filter.permute(0, 1, 3, 2)[:, :, 0::2, :] * h_sum) + torch.sum(self.high_pass_filter.permute(0, 1, 3, 2)[:, :, 1::2, :] * l_sum)
        weight2 = torch.sum(self.low_pass_filter.permute(0, 1, 3, 2)[:, :, 0::2, :] * l_sum) + torch.sum(self.low_pass_filter.permute(0, 1, 3, 2)[:, :, 1::2, :] * h_sum)
    
        # reconstruct ll + hl = l
        x_l_combine = self.switch_data(x[:, 0:channel // 4, :, :], x[:, channel // 4 : channel // 2, :, :], 2)
        x_l_combine = torch.nn.functional.pad(x_l_combine,
                                              pad = [0, 0, self.kernel_size // 2, self.kernel_size // 2],
                                              mode = 'reflect')
        x_l_e = torch.nn.functional.conv2d(x_l_combine,
                                         self.high_pass_filter.permute(0, 1, 3, 2).repeat(x_l_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (2, 1))
        x_l_o = torch.nn.functional.conv2d(x_l_combine[:, :, 1:, :],
                                         self.low_pass_filter.permute(0, 1, 3, 2).repeat(x_l_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (2, 1))

        x_l_e = x_l_e / weight1
        x_l_o = x_l_o / weight2
       
        x_l = self.switch_data(x_l_e, x_l_o, 2)
        
        # reconstruct lh + hh = h
        x_h_combine = self.switch_data(x[:, channel // 2 : channel // 4 * 3, :, :], x[:, channel // 4 * 3 : , :, :], 2)
        x_h_combine = torch.nn.functional.pad(x_h_combine,
                                              pad = [0, 0, self.kernel_size // 2, self.kernel_size // 2],
                                              mode = 'reflect')
        x_h_e = torch.nn.functional.conv2d(x_h_combine,
                                         self.high_pass_filter.permute(0, 1, 3, 2).repeat(x_h_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (2, 1))
        x_h_o = torch.nn.functional.conv2d(x_h_combine[:, :, 1:, :],
                                         self.low_pass_filter.permute(0, 1, 3, 2).repeat(x_h_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (2, 1))

        x_h_e = x_h_e / weight1
        x_h_o = x_h_o / weight2

        
        x_h = self.switch_data(x_h_e, x_h_o, 2)
        
        
        # reconstruct l + h = x
        x_combine = self.switch_data(x_l, x_h, 3)
        x_e = torch.nn.functional.conv2d(x_combine,
                                         self.high_pass_filter.repeat(x_h_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (1, 2),
                                         padding = [0, self.kernel_size // 2 if self.kernel_size != 2 else 0])
        x_o = torch.nn.functional.conv2d(x_combine[:, :, :, 1:],
                                         self.low_pass_filter.repeat(x_h_combine.shape[1], 1, 1, 1),
                                         groups = channel // 4,
                                         stride = (1, 2),
                                         padding = [0, self.kernel_size // 2 if self.kernel_size != 2 else 0])
        x_e = x_e / weight1
        x_o = x_o / weight2

        recover_x = self.switch_data(x_e, x_o, 3)
        
        return recover_x
    
    def forward(self, x):
        
        channel = x.shape[1]
        # pad the feature map
        x_pad = torch.nn.functional.pad(x,
                                    pad = [self.kernel_size // 2 if self.kernel_size != 2 else 0, self.kernel_size // 2 if self.kernel_size != 2 else 0, 0, 0],
                                    mode = 'reflect')
        # vertical side
        x_l = torch.nn.functional.conv2d(x_pad, 
                                         self.low_pass_filter.repeat(channel, 1, 1, 1),
                                         groups = channel, 
                                         stride = (1, self.stride))
        x_h = torch.nn.functional.conv2d(x_pad[:, :, :, 1:], 
                                         self.high_pass_filter.repeat(channel, 1, 1, 1),
                                         groups = channel, 
                                         stride = (1, self.stride))
        # pad the feature map
        x_l = torch.nn.functional.pad(x_l,
                                    pad = [0, 0, self.kernel_size // 2 if self.kernel_size != 2 else 0, self.kernel_size // 2 if self.kernel_size != 2 else 0],
                                    mode = 'reflect')
        x_h = torch.nn.functional.pad(x_h,
                                    pad = [0, 0, self.kernel_size // 2 if self.kernel_size != 2 else 0, self.kernel_size // 2 if self.kernel_size != 2 else 0],
                                    mode = 'reflect')
        # horizontal side
        x_ll = torch.nn.functional.conv2d(x_l, 
                                          self.low_pass_filter.permute(0, 1, 3, 2).repeat(channel, 1, 1, 1), 
                                          groups = x_l.shape[1], 
                                          stride = (self.stride, 1))
        x_hl = torch.nn.functional.conv2d(x_l[:, :, 1:, :], 
                                          self.high_pass_filter.permute(0, 1, 3, 2).repeat(channel, 1, 1, 1), 
                                          groups = x_l.shape[1], 
                                          stride = (self.stride, 1))
        x_lh = torch.nn.functional.conv2d(x_h, 
                                          self.low_pass_filter.permute(0, 1, 3, 2).repeat(channel, 1, 1, 1), 
                                          groups = x_h.shape[1], 
                                          stride = (self.stride, 1))
        x_hh = torch.nn.functional.conv2d(x_h[:, :, 1:, :], 
                                          self.high_pass_filter.permute(0, 1, 3, 2).repeat(channel, 1, 1, 1), 
                                          groups = x_h.shape[1], 
                                          stride = (self.stride, 1))
        del x_l
        del x_h
        
        x_all = torch.cat([x_ll, x_hl, x_lh, x_hh], dim = 1)
        return x_all
    
class Energy_attention(nn.Module):
    def __init__(self, in_cha):
        super(Energy_attention, self).__init__()
        self.batch_norm = nn.BatchNorm2d(in_cha)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.SE = nn.Sequential(nn.Linear(in_cha, in_cha // 4),
                                nn.BatchNorm1d(in_cha // 4),
                                nn.ReLU(inplace = True),
                                nn.Linear(in_cha // 4, in_cha),
                                nn.Sigmoid())
        
    def forward(self, x):
        x_norm = self.batch_norm(x)
        x_energy = self.avgpool(torch.pow(x_norm, 2)).squeeze(-1).squeeze(-1)
        x_energy = self.SE(x_energy)
        
        x = x * x_energy.unsqueeze(-1).unsqueeze(-1)
        
        return x


