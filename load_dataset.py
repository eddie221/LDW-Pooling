#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 29 16:15:05 2021

@author: mmplab603
"""

import torchvision.transforms as transforms
import torchvision
from PIL import Image
from config import IMAGE_SIZE, BATCH_SIZE, KFOLD
if KFOLD == 1:
    from config import VAL_SPLIT
import torch
from sklearn.model_selection import KFold
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import glob
from torch.utils.data.dataset import Dataset
import os

data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((300, 300), Image.BILINEAR),
            transforms.RandomCrop(IMAGE_SIZE),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
}

def load_data_cifar100(path):
    dataloader = []
    dataset_sizes = []
    trainset = torchvision.datasets.CIFAR100(root=path,
                                            train = True,
                                            download = True,
                                            transform = data_transforms['train'])
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size = BATCH_SIZE,
                                              shuffle = True,
                                              num_workers = 2)
    
    testset = torchvision.datasets.CIFAR100(root=path,
                                           train = False,
                                           download = True,
                                           transform = data_transforms['val'])
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size = BATCH_SIZE,
                                             shuffle = False,
                                             num_workers = 2)
    
    dataloader.append({'train' : trainloader, 'val' : testloader})
    dataset_sizes.append({'train' : len(trainloader), 'val' : len(testloader)})
    
    return dataloader, None

def load_data_cifar10(path):
    dataloader = []
    dataset_sizes = []
    trainset = torchvision.datasets.CIFAR10(root=path,
                                            train = True,
                                            download = True,
                                            transform = data_transforms['train'])
    trainloader = torch.utils.data.DataLoader(trainset,
                                              batch_size = BATCH_SIZE,
                                              shuffle = True,
                                              num_workers = 2)
    
    testset = torchvision.datasets.CIFAR10(root=path,
                                           train = False,
                                           download = True,
                                           transform = data_transforms['val'])
    testloader = torch.utils.data.DataLoader(testset,
                                             batch_size = BATCH_SIZE,
                                             shuffle = False,
                                             num_workers = 2)
    
    dataloader.append({'train' : trainloader, 'val' : testloader})
    dataset_sizes.append({'train' : len(trainloader), 'val' : len(testloader)})
    
    return dataloader, None


def load_data(path):
    all_image_datasets = torchvision.datasets.ImageFolder(path, data_transforms['train'])
    
    dataloader = []
    dataset_sizes = []
    if KFOLD != 1:
        kf = KFold(KFOLD, shuffle = True)
        for train_idx, val_idx in kf.split(all_image_datasets):
            train_dataset = torch.utils.data.Subset(all_image_datasets, train_idx)
            trainloader = torch.utils.data.DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)    
            val_dataset = torch.utils.data.Subset(all_image_datasets, val_idx)
            valloader = torch.utils.data.DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 2)
            dataloader.append({'train' : trainloader, 'val' : valloader})
            dataset_sizes.append({'train' : len(trainloader), 'val' : len(valloader)})
    else:
        indices = list(range(len(all_image_datasets)))
        dataset_size = len(all_image_datasets)
        split = int(np.floor(VAL_SPLIT * dataset_size))
        np.random.seed(0)
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        trainloader = torch.utils.data.DataLoader(all_image_datasets,
                                                       batch_size = BATCH_SIZE,
                                                       sampler = train_sampler,
                                                       num_workers = 16)
        valloader = torch.utils.data.DataLoader(all_image_datasets,
                                                       batch_size = BATCH_SIZE,
                                                       sampler = valid_sampler,
                                                       num_workers = 16)
        dataloader.append({'train' : trainloader, 'val' : valloader})
        dataset_sizes.append({'train' : len(trainloader), 'val' : len(valloader)})
        
        
    return dataloader, dataset_sizes, all_image_datasets
