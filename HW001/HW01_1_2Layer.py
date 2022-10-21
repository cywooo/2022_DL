import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = os.getcwd()+'\energy_efficiency_data.csv' #使CSV檔的位置隨著資料夾變動
Raw_data_DF = pd.read_csv(file_path) #利用PANDAS將檔案的資料吃進來


def pre_trea(Raw_data_DF):
    Orien_OH=pd.get_dummies(Raw_data_DF['Orientation'],prefix = 'Orientation') #pandas分離出 Orientation 做one hot  
    GlazAD_OH=pd.get_dummies(Raw_data_DF['Glazing Area Distribution'],prefix = 'Glazing Area Distribution') #pandas分離出 Glazing Area Distribution
    Raw_data_DF=Raw_data_DF.drop(['Orientation','Glazing Area Distribution'],axis=1)
    Raw_data_DF = pd.concat([Raw_data_DF,Orien_OH,GlazAD_OH],axis=1)
    
    data_train_DF =Raw_data_DF.sample(frac=0.75,random_state=np.random.randint(1e5),axis=0)
    data_test_DF =Raw_data_DF[~Raw_data_DF.index.isin(data_train_DF.index)]
    
    target_train_DF=data_train_DF.pop('Heating Load') #pandas分離出 Heating Load 做one hot  
    target_test_DF=data_test_DF.pop('Heating Load')
    #將資料轉為NP
    data_train=data_train_DF.values
    data_test=data_test_DF.values
    target_train=np.reshape(target_train_DF.values,[len(target_train_DF),1])
    target_test=np.reshape(target_test_DF.values,[len(target_test_DF),1])

    return data_train,data_test,target_train,target_test

def active(x):
    return x

def dactive(x):
    out= np.ones((len(x), 1))
    return out


np.random.seed()

#參數與矩陣設定
learning_rate = 1e-6
epoch = 300
batch = 1

features_train,features_test,target_train,target_test = pre_trea(Raw_data_DF) #CALL資料預處裡函式
valid_num = len(features_test)#分資料集
train_num = len(features_train)
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
    l1 = np.dot(x, w1).T + b1
    a1 = active(l1)
    l2 = np.dot(a1.T, w2) + b2
    a2 = active(l2)

    return l1, l2, a1, a2

#def back(x, n, w1, w2, w3, b1, b2, b3):
def back( feature,real, w1, w2, b1, b2, l1, l2, a1, a2, rl): 
    feature=np.reshape(feature,(1,len(feature)))
    real=np.reshape(real,(len(real),1))
    Edy=np.reshape(-2*(real-l2),(len(real),1))
    
    dw2=np.dot(l1,Edy*dactive(a2))
    dw1=np.dot(feature.T,(np.dot((w2*dactive(a1)),Edy*dactive(a2))).T)
    
    db2=Edy*dactive(a2)
    db1=np.dot((w2*dactive(a1)),Edy*dactive(a2))

    w1= w1-learning_rate*dw1
    w2= w2-learning_rate*dw2
    b1= b1-learning_rate*db1
    b2= b2-learning_rate*db2
    return w1, w2, b1, b2 ,dw1, dw2, db1, db2

def loss ( w1, w2, b1, b2, target, features):
    target=np.reshape(target,(len(target),1))
    _,_,a2,_ =foward(features, w1, w2, b1, b2)
    re=np.sqrt(np.average(np.power(a2.T-target,2)))
    return  re


for i in range(epoch):
    for j in range(batch_epoch):
        L1, L2, A1, A2 = foward(features_train[j], w1, w2, b1, b2)
        w1, w2, b1, b2 ,dw1, dw2, db1, db2= back(features_train[j],target_train[j], w1, w2, b1, b2, L1, L2, A1, A2, learning_rate)
    tloss_draw[i] = loss( w1, w2, b1, b2, target_test, features_test)
    vloss_draw[i] = loss( w1, w2, b1, b2, target_train, features_train)
    print('Training parameters: epochs = %d tloss = %f vloss = %f' % (i+1, tloss_draw[i], vloss_draw[i]))
    
plt.plot(np.linspace(1, epoch, epoch), tloss_draw, label='training')
plt.plot(np.linspace(1, epoch, epoch), vloss_draw, label='testting')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(title='learning curve :')
plt.show()












