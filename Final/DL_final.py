# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:53:41 2020

@author: USER
"""

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
from torchvision import transforms, datasets 
from torch.utils.data import DataLoader
import numpy as np
import csv

'''
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)
        #self.bn1 = nn.BatchNorm2d(planes,eps=0.001, momentum=0.99)
        self.bn1 = nn.BatchNorm2d(planes)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.prelu2 = nn.PReLU()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = self.prelu1(out)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.prelu2(out)
        return out
'''
class Recognition(nn.Module):
    '''
    def __init__(self,classes):
        super(Recognition, self).__init__()
        self.newresnet50 = nn.Sequential(*(list((resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)).children())[:-2]))

        %for param in self.newresnet50.parameters():
           % param.requires_grad = False

        self.bn1 = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.25)
        self.layer1 = nn.Sequential(nn.Conv2d(2048, 512, 5, padding=2),nn.BatchNorm2d(512),nn.ReLU(),nn.Dropout(0.25))
        self.layer2 = nn.Sequential(nn.Conv2d(512, 128, 3, padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.Dropout(0.25))
        self.layer3 = nn.Sequential(nn.Conv2d(128, 128, 3, padding=1),nn.BatchNorm2d(128),nn.ReLU(),nn.Dropout(0.25),nn.Flatten())
        self.fc1 = nn.Sequential(nn.Linear(128,256),nn.BatchNorm1d(256),nn.ReLU(),nn.Dropout(0.25))
        self.fc2 = nn.Sequential(nn.Linear(256,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.25))
        self.fc3 = nn.Linear(1024,classes)
        #self.fc3 = nn.Sequential(nn.Linear(512,7), nn.LogSoftmax(dim=1))

    def forward(self, x):
        #print(x.shape)
        out = self.newresnet50(x)
        out = self.dp(self.relu(self.bn1(out)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        #print(out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        #print(out.shape)
        return out
    '''
    def __init__(self,classes):
        super(Recognition, self).__init__()
        self.newresnet50 = nn.Sequential(*(list((resnet50(weights=ResNet152_Weights.IMAGENET1K_V2)).children())[:-2]))
        '''
        for param in self.newresnet50.parameters():
            param.requires_grad = False
        '''
        self.bn1 = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU()
        self.dp = nn.Dropout(0.25)
        self.fl = nn.Sequential(nn.Flatten())
        self.fc1 = nn.Sequential(nn.Linear(2048,1024),nn.BatchNorm1d(1024),nn.ReLU(),nn.Dropout(0.25))
        self.fc2 = nn.Linear(1024,classes)
        
    def forward(self, x):
        #print(x.shape)
        out = self.newresnet50(x)
        out = self.dp(self.relu(self.bn1(out)))
        #print(out.shape)
        out = self.fl(out)
        out = self.fc1(out)
        out = self.fc2(out)
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
         transforms.ToTensor(),
         transforms.Normalize(mean, std,inplace = True)
        ])
        
      
    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]
    )  
    dataset_train = datasets.CIFAR100(data_path_cifar, train=True, transform=transform_train, download=True)
    dataset_test = datasets.CIFAR100(data_path_cifar, train=False, transform=transform_test, download=True)
    classes = 100
    lr = 0.0001

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_train = DataLoader(dataset_train, batch_size=64,shuffle=True,num_workers=2)
    data_test = DataLoader(dataset_test, batch_size=64,shuffle=False,num_workers=2)
    recognition = Recognition(classes).cuda()
    optimizer = torch.optim.Adam(recognition.parameters(),lr=0.0001,weight_decay = 1e-4)
    loss_func = nn.CrossEntropyLoss().cuda()
    epoch = 200
    
    acc_begin = 0
    

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
        '''
        correct = 0
        total = 0
        recognition.eval()
        for idx, (data, target) in enumerate(data_test):
            data, target = data.to(device), target.to(device)
            y = recognition(data)
            test_loss = loss_func(y, target)
            _, predicted1 = torch.max(y.data, 1)
            total += target.size(0)
            correct += (predicted1 == target.data).sum().item()
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f})'.format(test_loss, correct, total, correct/total))
        '''
        accuracy = correct/total
        if accuracy>acc_begin:
            step = epoch
            torch.save(recognition.state_dict(), './model/recognition.pkl')
            acc_begin = accuracy
        
        if epoch ==99:
            lr /= 10
        elif epoch == 149:
            lr /= 10
        


    category=[]
    recognition.load_state_dict(torch.load("./model/recognition.pkl"),strict=True)
    correct = 0
    total = 0
    recognition.eval()
    for idx, (data, target) in enumerate(data_test):
        data, target = data.to(device), target.to(device)
        y = recognition(data)
        test_loss = loss_func(y, target)
        _, predicted1 = torch.max(y.data, 1)
        total += target.size(0)
        correct += (predicted1 == target.data).sum().item()
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(test_loss, correct, total, correct/total))
    category=np.concatenate(category)
    with open('team_7_submission2.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['ID', 'Category'])
        for i in range(1000):
            writer.writerow([i,category[i]])