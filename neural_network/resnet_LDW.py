#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 16:34:25 2021
@author: mmplab603
"""

import torch
import torch.nn as nn
from neural_network.LDW_pool import LDW_down, Energy_attention

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class BasicBlock_LDW(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock_LDW, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, planes // 4)
        self.bn1 = norm_layer(planes // 4)
        self.conv2 = conv3x3(planes // 4, planes // 4, stride)
        self.bn2 = norm_layer(planes // 4)
        self.relu = nn.ReLU(inplace=True)
        self.conv3 = conv3x3(planes // 4, planes // 4)
        self.bn3 = norm_layer(planes // 4)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        identity = x
        out = self.conv2(x)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out = out.repeat(1, 4, 1, 1)
        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class Bottleneck_LDW(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck_LDW, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width // 4)
        self.bn1 = norm_layer(width // 4)
        self.conv2 = conv3x3(width // 4, width // 4, stride, groups, dilation)
        self.bn2 = norm_layer(width // 4)
        self.conv3 = conv1x1(width // 4, planes // 4)
        self.bn3 = norm_layer(planes // 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        identity = out
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(identity)
            
        out = out.repeat(1, 4, 1, 1)
        out += identity
        out = self.relu(out)

        return out
    
class Resnet(nn.Module):
    def __init__(self, num_classes, layers, block, LDW_block):
        super(Resnet, self).__init__()
        self.name = "resnet_LDW"
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.LDW_pool = LDW_down(7, 2)
        self.lifting1 = nn.Sequential(self.LDW_pool,
                                      nn.Conv2d(256, 64, 1, bias = False))

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, LDW_block, 64, layers[0])
        self.layer2 = self._make_layer(block, LDW_block, 128, layers[1], stride = 2)
        self.layer3 = self._make_layer(block, LDW_block, 256, layers[2], stride = 2)
        self.layer4 = self._make_layer(block, LDW_block, 512, layers[3], stride = 2)
        
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = self._construct_fc_layer([num_classes], 512 * block.expansion)
        
        
    def _make_layer(self, block, LDW_block, planes, blocks, stride = 1):
        downsample = None
        layers = []
        if stride != 1 or self.inplanes != planes * block.expansion:
            if stride == 1:
                downsample = nn.Sequential(
                    nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(planes * block.expansion),
                )
                layers.append(block(self.inplanes, planes, stride, downsample))
            else:
                downsample = nn.Sequential(
                    self.LDW_pool,
                    Energy_attention(planes * block.expansion),
                    nn.BatchNorm2d(planes * block.expansion),
                )
                layers.append(LDW_block(self.inplanes, planes * block.expansion, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
        
    def _construct_fc_layer(self, fc_dims, input_dim, dropout_p=None):
        """
        Construct fully connected layer
        - fc_dims (list or tuple): dimensions of fc layers, if None,
                                   no fc layers are constructed
        - input_dim (int): input dimension
        - dropout_p (float): dropout probability, if None, dropout is unused
        """
        if fc_dims is None:
            self.feature_dim = input_dim
            return None
        
        assert isinstance(fc_dims, (list, tuple)), "fc_dims must be either list or tuple, but got {}".format(type(fc_dims))
        
        layers = []
        for dim in fc_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p is not None:
                layers.append(nn.Dropout(p=dropout_p))
            input_dim = dim
        
        self.feature_dim = fc_dims[-1]
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        if torch.isnan(x).any():
            print("input")
            print(torch.nonzero(torch.isnan(x)))
        x = self.conv1(x)
        if torch.isnan(x).any():
            print("conv1")
            print(x)
        x = self.lifting1(x)
        if torch.isnan(x).any():
            print("lift1")
        x = self.bn1(x)
        if torch.isnan(x).any():
            print("bn1")
        x = self.relu(x)
        if torch.isnan(x).any():
            print("relu")
        x = self.lifting1(x)
        if torch.isnan(x).any():
            print("lift2")
        #x = self.maxpool(x)
        # 64, 112, 112
        
        # layer1
        x = self.layer1(x)
        if torch.isnan(x).any():
            print("layer1")
        # layer2
        x = self.layer2(x)
        if torch.isnan(x).any():
            print("layer2")
        
        # layer3
        x = self.layer3(x)
        if torch.isnan(x).any():
            print("layer3")
        
        # layer4
        x = self.layer4(x)
        if torch.isnan(x).any():
            print("layer4")
        x = self.avg(x).view(x.shape[0], -1)
        
        x = self.fc(x)
        
        return x

def resnet18(num_classes = 1000):
    model = Resnet(num_classes, [2, 2, 2, 2], BasicBlock, BasicBlock_LDW)
    return model
    
def resnet50(num_classes = 1000):
    model = Resnet(num_classes, [3, 4, 6, 3], Bottleneck, Bottleneck_LDW)
    return model


def resnet101(num_classes = 1000):
    model = Resnet(num_classes, [3, 4, 23, 3], Bottleneck, Bottleneck_LDW)
    return model

if __name__ == "__main__":
    model = resnet50(10)
    a = torch.randn([2, 3, 224, 224])
    with open('../resnet18_lifting.txt', 'w') as f:
        print(model, file = f)
# =============================================================================
#     optim = torch.optim.Adam([
#         {'params' : [param for name, param in model.named_parameters() if name != "Lifting_down"]},
#         {'params' : [param for name, param in model.named_parameters() if name == "Lifting_down"], 'lr' : 1e-2},
#         ], lr = 1e-4, weight_decay = 1e-4)
# =============================================================================
# =============================================================================
#     param = torch.load('../pkl/fold_0_best_20210408-3.pkl')
#     model.load_state_dict(param)
# =============================================================================
    model(a)
# =============================================================================
#     optim = torch.optim.Adam(model.parameters(), lr = 1e-3)
#     loss_f = torch.nn.CrossEntropyLoss()
#     label = torch.tensor([0, 1])
#     for i in range(5):
#         output = model(a)
#         optim.zero_grad()
#         loss = loss_f(output, label) + model.LDW_pool.regular_term_loss()
#         print(loss)
#         model.LDW_pool.loss_cumulate = 0
#         loss.backward()
#         optim.step()
#         
#     model.eval()
#     pytorch_total_params = sum(p.numel() for p in model.parameters())
#     print(pytorch_total_params / 1000 / 1000)
# =============================================================================
        
    #model.lifting_pool[0].filter_constraint()
