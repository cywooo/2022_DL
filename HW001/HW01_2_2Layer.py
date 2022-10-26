import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = os.getcwd()+'\ionosphere_data.csv' #使CSV檔的位置隨著資料夾變動
Raw_data_DF = pd.read_csv(file_path,header=None) #利用PANDAS將檔案的資料吃進來
#Raw_data_DF=(Raw_data_DF-Raw_data_DF.mean())/Raw_data_DF.std()

def pre_trea(Raw_data_DF): 
    target_OH = pd.get_dummies(Raw_data_DF[34]) #pandas分離出 Orientation 做one hot
    Raw_data_DF=Raw_data_DF.drop(34,axis=1) #刪除舊資料
    Raw_data_DF = pd.concat([Raw_data_DF,target_OH],axis=1) #嫁接在一起
    data_train_DF =Raw_data_DF.sample(frac=0.8,random_state=np.random.randint(1e5),axis=0)
    data_test_DF =Raw_data_DF[~Raw_data_DF.index.isin(data_train_DF.index)]
    
    #會有error要改
    target_train_DF= pd.concat([data_train_DF.pop('b'),data_train_DF.pop('g')],axis=1)
    target_test_DF= pd.concat([data_test_DF.pop('b'),data_test_DF.pop('g')],axis=1) #Bad在前 Good在後

    #將資料轉為NP
    data_train=data_train_DF.values
    data_test=data_test_DF.values
    target_train=target_train_DF.values
    target_test=target_test_DF.values

    return data_train,data_test,target_train,target_test

#定義ReLU function
def ReLU(x):
    m, n = x.shape
    for i in range(m):
        for j in range(n):
            if x[i][j] < 0: 
                x[i][j] = 0
    return x

#定義deReLU function
def dReLU(x):
    m, n = x.shape
    for i in range(m):
        for j in range(n):
            if x[i][j] < 0:
                x[i][j] = 0
            else:
                x[i][j] = 1
    return x

#定義Softmax function
def Softmax(x):
    return (np.exp(x)/np.sum(np.exp(x),axis = 1 ,keepdims=True)).T
'''
    value = np.zeros((n, num_of_targets))
    for i in range(n):
        #max_ = np.max(x[i])
        #value[i] = np.exp(x[i] - max_) / np.sum(np.exp(x[i] - max_)) #stable softmax
        value[i] = np.exp(x[i]) / np.sum(np.exp(x[i]))
    return value.T'''

def dsoft(x):
    x=x.T
    y_ , x_ =(x).shape
    re = np.zeros((y_,x_))
    for i in range(y_):
        for j in range(x_):
            re[i][j] = (np.exp(x[i][j])/sum(np.exp(x[i]))) - np.exp(2*x[i][j])*np.power(sum(np.exp(x[i])),-2)
    return re

np.random.seed()

#參數與矩陣設定
learning_rate = 1e-4
epoch = 100
Num_of_hiden1 = 40
Num_of_outLayer = 2
batch = 1
num_of_features= 34
num_of_targets= 2

features_train,features_test,target_train,target_test = pre_trea(Raw_data_DF) #CALL資料預處裡函式
#分資料集
train_num = len(features_train)
valid_num = len(features_test)
batch_epoch = np.int64(train_num / batch)

w1 = np.random.randn(num_of_features, Num_of_hiden1)  * np.sqrt(1. / num_of_features)
w2 = np.random.randn(Num_of_hiden1, Num_of_outLayer) * np.sqrt(1. / Num_of_hiden1)
b1 = np.random.randn(Num_of_hiden1, 1) * np.sqrt(1. / Num_of_hiden1)
b2 = np.random.randn(Num_of_outLayer,1) * np.sqrt(1. / Num_of_outLayer)
tloss_draw=np.zeros(epoch)
vloss_draw=np.zeros(epoch)

#feed forward
def foward(x, w1, w2, b1, b2,n):
    x=np.reshape(x,(-1, num_of_features))
    a1 = np.dot(x, w1).T + b1
    l1 = ReLU(a1)
    a2 = np.dot(l1.T, w2).T + b2
    l2 = Softmax(a2.T)

    return l1, l2, a1, a2

#def back(x, n, w1, w2, w3, b1, b2, b3):
def back( feature,real, w1, w2, b1, b2, l1, l2, a1, a2, rl,n): 
    feature_=np.reshape(feature,(-1,num_of_features))
    real=np.reshape(real,(-1,num_of_targets))
    Edy=np.reshape((-1*real/l2.T)+((1-real)/(1-l2.T)),(-1,num_of_targets)) #-1*real/(l2.T*np.log(2)) #(-1*real/l2.T)+((1-real)/(1-l2.T))
    #Edy=np.reshape(-1*real/(l2.T*np.log(2)),(-1,num_of_targets))
    #Edy=np.reshape(real-a2.T,(-1,num_of_targets))
    
    dw2=np.dot(l1,Edy*dsoft(a2))
    dw1=np.dot(feature_.T,(np.dot((w2*dReLU(a1)),(Edy*dsoft(a2)).T)).T)
    db2=(Edy*dsoft(a2)).T
    db1=np.dot((w2*dReLU(a1)),(Edy*dsoft(a2)).T)

    w1= w1-learning_rate*dw1
    w2= w2-learning_rate*dw2
    b1= b1-learning_rate*db1
    b2= b2-learning_rate*db2
    return w1, w2, b1, b2

def cost ( w1, w2, b1, b2, target, features):
    l1, l2, a1, a2 =foward(features, w1, w2, b1, b2,len(target))
    ef = np.zeros((len(target), num_of_targets))
    l2=l2.T
    sum_ = 0
    for i in range(len(target)):
        for j in range(num_of_targets):
            if l2[i][j] > 0.00001:
                ef[i][j] = -1*(target[i][j]) * (np.log(l2[i][j])/np.log(2))
            if (ef[i][j]) != 0:
                sum_ = sum_+ ef[i][j]
    return (sum_ / len(target))


def scatter_draw(a_out,y):
    y=y/np.amax(y ,axis=0,keepdims=True)
    a_out=np.concatenate((a_out,y), axis=0)
    class_1=a_out[:,a_out[2,:]==1]
    class_2=a_out[:,a_out[2,:]!=1]
    plt.scatter(class_1[0,:],class_1[1,:],marker='+')
    plt.scatter(class_2[0,:],class_2[1,:],marker='o')
    plt.show()
    
def batch_pick(n,train_num,counter):
    if(n==1):
        return features_train[counter] , target_train[counter]
    else:
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
        L1, L2, A1, A2 = foward(features_train_batch, w1, w2, b1, b2, batch)
        w1, w2, b1, b2= back(features_train_batch,target_train_batch,w1, w2, b1, b2, L1, L2, A1, A2, learning_rate, batch)
    vloss_draw[i] = cost( w1, w2, b1, b2, target_test, features_test)
    tloss_draw[i] = cost( w1, w2, b1, b2, target_train, features_train)
    print('Training parameters: epochs = %d tloss = %f vloss = %f' % (i+1, tloss_draw[i], vloss_draw[i]))
  

plt.plot(np.linspace(1, epoch, epoch), tloss_draw, label='training')
plt.plot(np.linspace(1, epoch, epoch), vloss_draw, label='testing')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend(title='learning curve :')
plt.show()

l1_, l2_, a1_, a2_ =foward(features_train, w1, w2, b1, b2, batch)
scatter_draw(a2_,l2_)

'''
plt.scatter(, label='training')
plt.plot(np.linspace(1, epoch, epoch), vloss_draw, label='testing')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend(title='learning curve :')
plt.show()

plt.plot(np.linspace(1, epoch, epoch), tloss_draw, label='training')
plt.plot(np.linspace(1, epoch, epoch), vloss_draw, label='testing')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.legend(title='learning curve :')
plt.show()
'''
















