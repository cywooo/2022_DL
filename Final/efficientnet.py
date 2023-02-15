# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 19:07:54 2023

@author: USER
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.efficientnet import EfficientNet_V2_M_Weights
from torchvision import transforms, datasets 
from torch.utils.data import DataLoader
import numpy as np
import csv
from torchsummary import summary



class Recognition(nn.Module):
    def __init__(self, classes):
        super(Recognition, self).__init__()
        #self.eff = EfficientNet.from_pretrained('efficientnet-b0')
        self.eff = models.efficientnet_v2_m(weights=(EfficientNet_V2_M_Weights.IMAGENET1K_V1))
        #self.eff._fc = torch.nn.Linear(in_features=self.eff._fc.in_features, out_features=classes, bias=True)
        self.eff.classifier[1] = torch.nn.Linear(in_features=self.eff.classifier[1].in_features, out_features=classes, bias=True)
        '''
        for params in self.eff.parameters():
            params.requires_grad = False#nn.Sequential(*(list(EfficientNet.from_pretrained('efficientnet-b7').children())[:-1]))
        #self.eff2 = nn.Sequential(*(list((self.eff).children())[:-2]))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(1000,classes))
        '''
        #self.fc3 = nn.Sequential(nn.Linear(512,7), nn.LogSoftmax(dim=1))

    def forward(self, x):
        out = self.eff(x)
        #print(out.shape)
        #out = self.fc(out)
        #print(out.shape)
        return out


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
                param_group['lr'] = lr
             
if __name__ == '__main__':
    data_path_cifar = './cifar-100';
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    transform_train = transforms.Compose(
        [
            
         transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
         transforms.RandomHorizontalFlip(p=0.5),
         transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean, std,inplace = True)
        ])
        
      
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.Normalize(mean, std)
        ]
    )  
    dataset_train = datasets.CIFAR100(data_path_cifar, train=True, transform=transform_train, download=True)
    dataset_test = datasets.CIFAR100(data_path_cifar, train=False, transform=transform_test, download=True)
    classes = 100
    lr = 0.0001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #data_train = DataLoader(dataset_train, batch_size=16,shuffle=True,num_workers=2)
    data_test = DataLoader(dataset_test, batch_size=16,shuffle=False,num_workers=2)
    recognition = Recognition(classes).cuda()
    optimizer = torch.optim.Adam(recognition.parameters(),lr = 0.0001,weight_decay = 1e-4)
    loss_func = nn.CrossEntropyLoss().cuda()
    
    epoch = 50
    weight_decay = 1e-4
    acc_begin = 0
    print("model finished")
    #effv2_m = models.efficientnet_v2_m(weights=(EfficientNet_V2_M_Weights.IMAGENET1K_V1))
    #summary(effv2_m.cuda(), (3,224,224))
    #eff = EfficientNet.from_pretrained('efficientnet-b0')
    #eff ._fc= torch.nn.Linear(in_features=eff._fc.in_features, out_features=100, bias=True)
    #summary(recognition, (3,224,224))
    
    '''
    for i in range(epoch):
        adjust_learning_rate(optimizer, lr)
        train_loss = 0
        correct = 0
        total = 0
        recognition.train()
        print(i)
        for idx, (data, target) in enumerate(data_train):
            data, target = data.to(device), target.to(device)
            y=recognition(data)
            optimizer.zero_grad()           
            train_loss = loss_func(y, target)
            train_loss.backward()                 # backpropagation, compute gradients   
            optimizer.step()                # apply gradients
            _, predicted1 = torch.max(y.data, 1)
            total += target.size(0)
            correct += (predicted1 == target.data).sum().item()
        print('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f})'.format(train_loss, correct, total, correct/total))
        
        accuracy = correct/total
        if accuracy>acc_begin:
            step = epoch
            torch.save(recognition.state_dict(), './model/recognition_effv2_m.pkl')
            acc_begin = accuracy
        
        if i ==14:
            lr /= 10
        elif i == 29:
            lr /= 10
        

   
    '''
    category=[]
    recognition.load_state_dict(torch.load("./model/recognition_effv2_m.pkl"),strict=True)
    correct = 0
    total = 0
    recognition.eval()
    for idx, (data, target) in enumerate(data_test):
        data, target = data.to(device), target.to(device)
        y = recognition(data)
        test_loss = loss_func(y, target)
        _, predicted1 = torch.max(y.data, 1)
        category.append(np.array(predicted1.cpu()))
        total += target.size(0)
        correct += (predicted1 == target.data).sum().item()
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, total, correct/total))
    category=np.concatenate(category)
    with open('team_7_submission_effv2_m.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Category'])
        for i in range(10000):
            writer.writerow([i,category[i]])
   
  