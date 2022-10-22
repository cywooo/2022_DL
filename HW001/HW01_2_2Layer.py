import pandas as pd
import numpy as np
import random  
import matplotlib.pyplot as plt
import os

file_path = os.getcwd()+'\ionosphere_data.csv' #使CSV檔的位置隨著資料夾變動
Raw_data_DF = pd.read_csv(file_path,header=None) #利用PANDAS將檔案的資料吃進來
#Raw_data_DF=(Raw_data_DF-Raw_data_DF.mean())/Raw_data_DF.std()

def pre_trea(Raw_data_DF): 
    data_train_DF =Raw_data_DF.sample(frac=0.8,random_state=np.random.randint(1e5),axis=0)
    data_test_DF =Raw_data_DF[~Raw_data_DF.index.isin(data_train_DF.index)]
    
    target_train_DF=data_train_DF.pop(34) #pandas分離出 Heating Load 做one hot  
    target_test_DF=data_test_DF.pop(34)
    target_test_DF=target_test_DF.get_dummies
    #將資料轉為NP
    data_train=data_train_DF.values
    data_test=data_test_DF.values
    target_train=np.reshape(target_train_DF.values,[len(target_train_DF),1])
    target_test=np.reshape(target_test_DF.values,[len(target_test_DF),1])

    return data_train,data_test,target_train,target_test

def active(x):
    return x

def dactive(x):
    out= np.ones((len(x),1))
    return out


np.random.seed(10)

#參數與矩陣設定
learning_rate = 1e-9
epoch = 300
batch = 1

features_train,features_test,target_train,target_test = pre_trea(Raw_data_DF) #CALL資料預處裡函式
#分資料集
train_num = len(features_train)
valid_num = len(features_test)
num_of_features= len(features_train.T)
num_of_targets= len(target_train.T)
batch_epoch = np.int64(train_num / batch)

w1 = np.random.randn(17, 16)  * np.sqrt(1. / 17)
w2 = np.random.randn(16, 1) * np.sqrt(1. / 16)
b1 = np.random.randn(16,1) * np.sqrt(1. / 16)
b2 = np.random.randn(1,1) * np.sqrt(1. / 1)
tloss_draw=np.zeros(epoch)
vloss_draw=np.zeros(epoch)

#feed forward
def foward(x, w1, w2, b1, b2):
    x=np.reshape(x,(-1, num_of_features))
    a1 = np.dot(x, w1).T + b1
    l1 = active(a1)
    a2 = np.dot(l1.T, w2) + b2
    l2 = active(a2)

    return l1, l2, a1, a2

#def back(x, n, w1, w2, w3, b1, b2, b3):
def back( feature,real, w1, w2, b1, b2, l1, l2, a1, a2, rl,n): 
    feature_=np.reshape(feature,(1,num_of_features))
    real_=np.reshape(real,(1,num_of_targets))
    Edy=np.reshape(-2*(real_-l2),(1,num_of_targets))
        
    dw2=np.dot(l1,Edy.T*dactive(a2))
    dw1=np.dot(feature_.T,(np.dot((w2*dactive(a1)),Edy.T*dactive(a2))).T)
    db2=Edy.T*dactive(a2)
    db1=np.dot((w2*dactive(a1)),Edy.T*dactive(a2))

    w1= w1-learning_rate*dw1
    w2= w2-learning_rate*dw2
    b1= b1-learning_rate*db1
    b2= b2-learning_rate*db2
    return w1, w2, b1, b2

def loss ( w1, w2, b1, b2, target, features):
    target=np.reshape(target,(-1,num_of_targets))
    l1, l2, a1, a2 =foward(features, w1, w2, b1, b2)
    
    re=np.sqrt(np.average(np.power(l2-target,2)))
    return  re

def batch_pick(n,train_num,counter):
    if(n==1):
        return features_train[counter] , target_train[counter]
    pick = range(counter*n,(counter+1)*n-1)
    features_train_batch =  np.zeros((n, num_of_features))
    target_train_batch = np.zeros((n, num_of_targets))
    for i in range(n):
        features_train_batch[i] = features_train[pick[i]]
        target_train_batch[i] = target_train[pick[i]]
    return features_train_batch, target_train_batch

for i in range(epoch):
    for j in range(batch_epoch):   
        features_train_batch,target_train_batch = batch_pick(batch ,train_num , j)
        L1, L2, A1, A2 = foward(features_train_batch, w1, w2, b1, b2)
        w1, w2, b1, b2= back(features_train_batch,target_train_batch,w1, w2, b1, b2, L1, L2, A1, A2, learning_rate, batch)
    tloss_draw[i] = loss( w1, w2, b1, b2, target_test, features_test)
    vloss_draw[i] = loss( w1, w2, b1, b2, target_train, features_train)
    print('Training parameters: epochs = %d tloss = %f vloss = %f' % (i+1, tloss_draw[i], vloss_draw[i]))
    
plt.plot(np.linspace(1, epoch, epoch), tloss_draw, label='training')
plt.plot(np.linspace(1, epoch, epoch), vloss_draw, label='testting')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(title='learning curve :')
plt.show()

_, l2_, _, _ = foward(features_train, w1, w2, b1, b2)
plt.plot(np.linspace(1, train_num, train_num), target_train, label='real')
plt.plot(np.linspace(1, train_num, train_num), l2_, label='predict')
plt.xlabel('# of Data')
plt.title('compare')
plt.legend(title='learning curve :')
plt.show()

_, l2_, _, _ = foward(features_test, w1, w2, b1, b2)
plt.plot(np.linspace(1, valid_num, valid_num), target_test, label='real')
plt.plot(np.linspace(1, valid_num, valid_num), l2_, label='predict')
plt.xlabel('# of Data')
plt.title('compare')
plt.legend(title='learning curve :')
plt.show()
















