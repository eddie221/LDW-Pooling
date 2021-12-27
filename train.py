#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 09:54:05 2020

@author: eddie
"""

import neural_network.resnet_LDW as resnet_LDW
import torch
import random

import numpy as np
import time
from config import *
import tqdm
import os
import logging
from load_dataset import load_data_cifar100, load_data_cifar10

if not os.path.exists('./pkl/{}/'.format(INDEX)):
    os.mkdir('./pkl/{}/'.format(INDEX))
#from torch.utils.tensorboard import SummaryWriter

#print environment information
print(torch.cuda.is_available())
DEVICE = 'cuda:0'

#writer = SummaryWriter('../tensorflow/logs/cub_{}'.format(INDEX), comment = "224_64")

use_gpu = torch.cuda.is_available()

optimizer_select = ''
loss_function_select = ''
model_name = ''
data_name = 'cifar10'

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

# =============================================================================
#     dataset_sizes = len(all_image_datasets)ting(256, kernel_size=
#     print(dataset_sizes)
#     
#     count = torch.zeros([NUM_CLASS])
#     
#     for data in all_image_datasets.imgs:
#         count[data[1]] += 1
#         
#     count = dataset_sizes['train'] / count
#     
#     count = count / count.sum()
#     print(count)
#     return all_image_dataloader, dataset_sizes, count
# =============================================================================

def create_nn_model():
    global model_name
    model_name = "resnet_LDW"
    #model = resnet.resnet18(num_classes = NUM_CLASS).to(DEVICE)
    model = resnet_LDW.resnet50(num_classes = NUM_CLASS).to(DEVICE)
    #model = LDW_CNN.LDW_CNN(num_classes = NUM_CLASS).to(DEVICE)
    #model = vgg_LDW.vgg16_bn(num_classes = NUM_CLASS).to(DEVICE)
    assert model_name == model.name, "Wrong model loading. Expect {} but get {}.".format(model_name, model.name)

    print(model)
    return model

def create_opt_loss(model):
    global optimizer_select
    global loss_function_select
    optimizer = [#torch.optim.SGD(model.parameters(), lr = LR, momentum = 0.9, weight_decay = 5e-4),
                 torch.optim.Adam(model.parameters(), lr = LR, weight_decay = 1e-4)
# =============================================================================
#                  torch.optim.Adam([{'params' : [param for name, param in model.named_parameters() if name != 'Lifting_down']},
#                                    {'params' : [param for name, param in model.named_parameters() if name == 'Lifting_down'], 'lr' : 1e-3}],
#                                   lr = LR, weight_decay = 1e-4)
# =============================================================================
                ]
    set_lr_secheduler = [torch.optim.lr_scheduler.MultiStepLR(optimizer[0], milestones=[100, 200, 300], gamma=0.1),
                        ]
    
    loss_func = [torch.nn.CrossEntropyLoss(),
                 torch.nn.MSELoss()]
    optimizer_select = 'Adam'
    loss_function_select = 'crossentropy'
    return optimizer, set_lr_secheduler, loss_func

def load_param(model):
    # load resnet
    params = torch.load("./pkl/resnet50.pkl", map_location = {"cuda:1" : "cuda:0"})
    load = []
    not_load = []
    for name, param in params.items():
        if name in model.state_dict():
            try:
                model.state_dict()[name].copy_(param)
                load.append(name)
            except:
                not_load.append(name)
        else:
            not_load.append(name)
    print("Load {} layers".format(len(load)))
    print("Not load {} layers".format(len(not_load)))
# =============================================================================
#     load = []
#     not_load = []
#     for name, param in params.items():
#         if name in model.backbone1.state_dict():
#             try:
#                 model.backbone1.state_dict()[name].copy_(param)
#                 load.append(name)
#             except:
#                 not_load.append(name)
#         else:
#             print(name)
#                 
#     print("Load {} layers".format(len(load)))
#     print("Not load {} layers".format(len(not_load)))
#     load = []
#     not_load = []
#     params = torch.load("../pretrain/resnet50.pth")
#     for name, param in params.items():
#         if name in model.backbone2.state_dict():
#             try:
#                 model.backbone2.state_dict()[name].copy_(param)
#                 load.append(name)
#             except:
#                 not_load.append(name)
#         else:
#             print(name)
#             
#     print("Load {} layers".format(len(load)))
#     print("Not load {} layers".format(len(not_load)))
# =============================================================================
            
    return model

def train_step(model, data, label, loss_func, optimizers, phase):
    if use_gpu:
        b_data = data.to(DEVICE)
        b_label = label.to(DEVICE)
    else:
        b_data = data
        b_label = label
    
    for optimizer in optimizers:
        optimizer.zero_grad() 
    output_1 = model(b_data)
    _, predicted = torch.max(output_1.data, 1)
    _, predicted5 = torch.topk(output_1.data, 5, dim = 1)  
    #print("output : ", output_1)
    #print("predicted : ", predicted)
    #print("label : ", label)
    #loss function
    cls_loss = loss_func[0](output_1, b_label)# + loss_func[0](output_1[1], b_label) + loss_func[0](output_1[2], b_label) + loss_func[0](output_1[3], b_label)
    filter_constraint = model.LDW_pool.regular_term_loss()
    loss = cls_loss + filter_constraint
    
    if phase == 'train':
        loss.backward()
        for optimizer in optimizers:
            optimizer.step()
            
    return loss.item(), predicted.detach().cpu(), predicted5.detach().cpu(), filter_constraint.detach().cpu(), cls_loss.detach().cpu()

#training
def training(job):
    global t_min_loss
    global optimizer_select
    global loss_function_select
    global model_name
    #with torch.autograd.set_detect_anomaly(True):
    #kfold_image_data, dataset_sizes, all_image_datasets = load_data()
    #kfold_image_data, all_image_datasets = load_data_cifar("./data")
    if data_name == 'cifar100':
        kfold_image_data, all_image_datasets = load_data_cifar100("./data")
    elif data_name == 'cifar10':
        kfold_image_data, all_image_datasets = load_data_cifar10("./data")
    #kfold_image_data, all_image_datasets = load_ImageNet("./imagenet")
    ACCMeters = []
    LOSSMeters = []
    for i in range(KFOLD):
        ACCMeters.append(AverageMeter(True))
        LOSSMeters.append(AverageMeter(False))
        
    for index, image_data in enumerate(kfold_image_data):
        model = create_nn_model()
        if PRETRAIN:
            print("Load pretrained")
            model = load_param(model)
        else:
            print("Not load pretrained")
        optimizers, lr_schedulers, loss_func = create_opt_loss(model)
        max_acc = {'train' : AverageMeter(True), 'val' : AverageMeter(True)}
        max_acc5 = {'train' : AverageMeter(True), 'val' : AverageMeter(True)}
        min_loss = {'train' : AverageMeter(False), 'val' : AverageMeter(False)}
        last_acc = {'train' : AverageMeter(True), 'val' : AverageMeter(True)}
        last_acc5 = {'train' : AverageMeter(True), 'val' : AverageMeter(True)}
        
        for epoch in range(1, EPOCH + 1):
            start = time.time()
            print('Fold {}/{} Epoch {}/{}'.format(index + 1, KFOLD, epoch, EPOCH))
            logging.info("-" * 15)
            logging.info('Fold {}/{} Epoch {}/{}'.format(index + 1, KFOLD, epoch, EPOCH))
            print('-' * 10)
            if CON_MATRIX:
                confusion_matrix = {'train' : np.zeros([NUM_CLASS, NUM_CLASS]), 'val' : np.zeros([NUM_CLASS, NUM_CLASS])}
            for phase in job:
                loss_t = AverageMeter(False)
                correct_t = AverageMeter(True)
                correct_t5 = AverageMeter(True)
                cls_rate = AverageMeter(False)
                con_rate = AverageMeter(False)
                
                if phase == 'train':
                    model.train(True)
                    if all_image_datasets is not None:
                        all_image_datasets.transform = data_transforms['train']
                else:
                    model.train(False)
                    if all_image_datasets is not None:
                        all_image_datasets.transform = data_transforms['val']
                step = 0
                for data, label in tqdm.tqdm(image_data[phase]):
                    loss, predicted, predicted5, constraint, cls_l = train_step(model, data, label, loss_func, optimizers, phase)
                    
                    cls_rate.update(cls_l, data.size(0))
                    con_rate.update(constraint, data.size(0))
                    loss_t.update(loss, data.size(0))
                    correct_t.update((predicted == label).sum().item(), label.shape[0])
                    correct_t5.update((predicted5 == label.unsqueeze(1)).sum().item(), label.shape[0])
                    step += 1
                    if CON_MATRIX:
                        np.add.at(confusion_matrix[phase], tuple([predicted.cpu().numpy(), label.detach().numpy()]), 1)
                
                if max_acc5[phase].avg < correct_t5.avg:
                    last_acc5[phase] = max_acc5[phase]
                    max_acc5[phase] = correct_t5
                    
                    if phase == 'val':
                        save_data = model.state_dict()
                        print('save')
                        torch.save(save_data, './pkl/{}/fold_{}_best5_{}.pkl'.format(INDEX, index, INDEX))
                        
                
                if max_acc[phase].avg < correct_t.avg:
                    last_acc[phase] = max_acc[phase]
                    max_acc[phase] = correct_t
                    
                    if phase == 'val':
                        ACCMeters[index] = correct_t
                        LOSSMeters[index] = loss_t
                        save_data = model.state_dict()
                        print('save')
                        torch.save(save_data, './pkl/{}/fold_{}_best_{}.pkl'.format(INDEX, index, INDEX))
                        
                logging.info("{} set loss : {:.6f}".format(phase, loss_t.avg))        
                logging.info("{} set acc : {:.6f}%".format(phase, correct_t.avg * 100.))        
                print('Index : {}'.format(INDEX))
                print("dataset : {}".format(data_name))
                print("Model name : {}".format(model_name))
                print("{} set loss : {:.6f}".format(phase, loss_t.avg))
                print("{} set CE : {:.6f}".format(phase, cls_rate.avg))
                print("{} set Constraint : {:.6f}".format(phase, con_rate.avg))
                #print("{} set cls : {:.6f}".format(phase, cls_rate_1.avg))
                #print("{} set min loss : {:.6f}".format(phase, min_loss[phase].avg))
                print("{} set acc : {:.6f}%".format(phase, correct_t.avg * 100.))
                print("{} last update : {:.6f}%".format(phase, (max_acc[phase].avg - last_acc[phase].avg) * 100.))
                print("{} set max acc : {:.6f}%".format(phase, max_acc[phase].avg * 100.))
                print("{} last update(5) : {:.6f}%".format(phase, (max_acc5[phase].avg - last_acc5[phase].avg) * 100.))
                print("{} set max acc(5) : {:.6f}%".format(phase, max_acc5[phase].avg * 100.))
                if CON_MATRIX:
                    print("{} confusion matrix :".format(phase))
                    print(confusion_matrix[phase])
                print()   
            
            for lr_scheduler in lr_schedulers:
                lr_scheduler.step()
                
            print(time.time() - start)
        del model
        del optimizers
        del lr_schedulers
        del loss_func
    acc = 0
    loss = 0
    for idx in range(1, len(ACCMeters) + 1):
        print("Fold {} best acc : {:.6f} loss : {:.6f}".format(idx, ACCMeters[idx - 1].avg, LOSSMeters[idx - 1].avg))
        acc += ACCMeters[idx - 1].avg
        loss += LOSSMeters[idx - 1].avg
    print("Avg. ACC : {:.6f} Avg. Loss : {:.6f}".format(acc / KFOLD, loss / KFOLD))
    
class AverageMeter():
    """Computes and stores the average and current value"""

    def __init__(self, acc):
        self.reset()
        self.acc = acc
    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        if self.acc:
            self.sum += value
        else:       
            self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count

        
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

if __name__ == '__main__':
    logging.basicConfig(filename = './pkl/{}/logging.txt'.format(INDEX), level=logging.DEBUG)
    logging.info('Index : {}'.format(INDEX))
    logging.info("dataset : {}".format(data_name))
    training = training(['val'])
    #dataloader, _ = load_ImageNet("../../dataset/imagenet")
    
