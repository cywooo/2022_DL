import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = os.getcwd()+'\energy_efficiency_data.csv' #使CSV檔的位置隨著資料夾變動
Raw_data_DF = pd.read_csv(file_path) #利用PANDAS將檔案的資料吃進來
#Raw_data_DF=(Raw_data_DF-Raw_data_DF.mean())/Raw_data_DF.std()

###資料預處理區(在Pandas的DF格式下進行one hot 以及 training&testing分群)
def pre_trea(Raw_data_DF):
    Orien_OH=pd.get_dummies(Raw_data_DF['Orientation'],prefix = 'Orientation') #pandas分離出 Orientation 做one hot  
    GlazAD_OH=pd.get_dummies(Raw_data_DF['Glazing Area Distribution'],prefix = 'Glazing Area Distribution') #pandas分離出 Glazing Area Distribution 做one hot  
    Raw_data_DF=Raw_data_DF.drop(['Orientation','Glazing Area Distribution'],axis=1) #刪除舊資料
    Raw_data_DF = pd.concat([Raw_data_DF,Orien_OH,GlazAD_OH],axis=1) #嫁接在一起
    
    data_train_DF =Raw_data_DF.sample(frac=0.75,random_state=np.random.randint(1e5),axis=0) #training&testing分群
    data_test_DF =Raw_data_DF[~Raw_data_DF.index.isin(data_train_DF.index)]
    
    target_train_DF=data_train_DF.pop('Heating Load')   #pandas分離出 Heating Load 做one hot  
    target_test_DF=data_test_DF.pop('Heating Load')
    data_train=data_train_DF.values                     #將資料轉為NP
    data_test=data_test_DF.values
    target_train=np.reshape(target_train_DF.values,[len(target_train_DF),1])
    target_test=np.reshape(target_test_DF.values,[len(target_test_DF),1])

    return data_train,data_test,target_train,target_test

#參數與矩陣設定
np.random.seed(1218) #1218 不錯
learning_rate = 1e-10 #1e-10 
epoch = 500 #500
batch = 10  #10

Num_of_hiden1 = 20 #20
Num_of_hiden2 = 10 #10
Num_of_outLayer = 1 #1

#CALL資料預處裡函式
features_train,features_test,target_train,target_test = pre_trea(Raw_data_DF) 

#常數擷取區:訓練與測試的資料筆數、特徵與目標值得個數、epoch內迴圈次數
train_num = len(features_train)
valid_num = len(features_test)
num_of_features= len(features_train.T)
num_of_targets= len(target_train.T)
batch_epoch = np.int64(train_num / batch)

#矩陣設置
w1 = np.random.randn(num_of_features, Num_of_hiden1) * np.sqrt(1. / num_of_features)
w2 = np.random.randn(Num_of_hiden1, Num_of_hiden2)  * np.sqrt(1. / Num_of_hiden1)
w3 = np.random.randn(Num_of_hiden2, Num_of_outLayer)  * np.sqrt(1. / Num_of_hiden2)
b1 = np.random.randn(Num_of_hiden1,1)    * np.sqrt(1. / Num_of_hiden1)
b2 = np.random.randn(Num_of_hiden2,1)   * np.sqrt(1. / Num_of_hiden2)
b3 = np.random.randn(Num_of_outLayer,1) * np.sqrt(1. / Num_of_outLayer)
tloss_draw=np.zeros(epoch)
vloss_draw=np.zeros(epoch)

#active function
def active(x):
    return x

#active function differential
def dactive(x):
    y,x =x.shape
    out= np.ones((y,x))
    return out

#feed forward
def foward(x, w1, w2, w3, b1, b2, b3):
    a1 = np.dot(w1.T, x.T) + b1
    l1 = active(a1)
    a2 = np.dot(w2.T, l1) + b2
    l2 = active(a2)
    a3 = np.dot(w3.T, l2) + b3
    l3 = active(a3)
    return l1, l2, l3, a1, a2, a3

#def back(x, n, w1, w2, w3, b1, b2, b3):
def back( x, t, w1, w2, w3, b1, b2, b3, l1, l2, l3, a1, a2, a3, rl, n): 
    Edy=-2*(t.T-l3)
    dw3 =np.dot( l2,  (Edy*dactive(a3)).T)
    dw2 =np.dot( l1,  (np.dot( w3, (Edy*dactive(a3)) )*dactive(a2)).T )    
    dw1 =np.dot( x.T, (np.dot( w2, ((np.dot( w3, (Edy*dactive(a3)) )*dactive(a2))) )*dactive(a1)).T)
    db3 =Edy*dactive(a3)
    db2 =np.dot( w3, Edy*dactive(a3))*dactive(a2)
    db1 =np.dot( w2,np.dot( w3, Edy*dactive(a3))*dactive(a2))*dactive(a1)
    w1= w1-rl*dw1
    w2= w2-rl*dw2
    w3= w3-rl*dw3
    b1= b1-np.sum(rl*db1, axis = 1,keepdims=True)
    b2= b2-np.sum(rl*db2, axis = 1,keepdims=True)
    b3= b3-np.sum(rl*db3, axis = 1,keepdims=True)
    return w1, w2, w3, b1, b2, b3

#to get lossing function
def loss ( w1, w2, w3, b1, b2, b3, t, x):
    _, _, l3, _, _, _=foward(x, w1, w2, w3, b1, b2, b3)  
    re=np.sqrt(np.average(np.power(l3-t,2)))
    return  re

#pick arrays as batch
def batch_pick( n, train_num, counter):
    features_train_batch = np.zeros((n, num_of_features))
    target_train_batch = np.zeros((n, num_of_targets))
    if((counter+n)>(train_num-1)):
        return features_train[counter:][:] , target_train[counter:][:]
    features_train_batch = features_train[counter:counter+n][:]
    target_train_batch = target_train[counter:counter+n][:]
    return features_train_batch, target_train_batch
 
for i in range(epoch):
    for j in range(batch_epoch):   
        features_train_batch,target_train_batch = batch_pick(batch ,train_num , j)
        L1, L2, L3, A1, A2, A3 = foward(features_train_batch, w1, w2, w3, b1, b2, b3)
        w1, w2, w3, b1, b2, b3 = back(features_train_batch,target_train_batch,w1, w2, w3, b1, b2, b3, L1, L2, L3, A1, A2, A3, learning_rate, batch)
    tloss_draw[i] = loss( w1, w2, w3, b1, b2, b3 , target_test, features_test)
    vloss_draw[i] = loss( w1, w2, w3, b1, b2, b3 , target_train, features_train)
    #print('Training parameters: epochs = %d tloss = %f vloss = %f' % (i+1, tloss_draw[i], vloss_draw[i]))
    
plt.plot(np.linspace(1, epoch, epoch), tloss_draw, label='training')
plt.plot(np.linspace(1, epoch, epoch), vloss_draw, label='testting')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('learning curve chart \n b= %d  epoch= %d Lr= %.1e\n final training loss %.3f ,testing loss %.3f'%(batch,epoch,learning_rate,tloss_draw[-1], vloss_draw[-1]))
plt.legend(title='learning curve :')
plt.show()

_, _, L3_, _, _, _ = foward(features_train, w1, w2, w3, b1, b2, b3)
plt.plot(np.linspace(1, train_num, train_num), target_train, label='real')
plt.plot(np.linspace(1, train_num, train_num), L3_.T, label='predict')
plt.xlabel('# of Data')
plt.title('training compare')
plt.legend(title='learning curve :')
plt.show()

_, _, L3_, _, _, _ = foward(features_test, w1, w2, w3, b1, b2, b3)
plt.plot(np.linspace(1, valid_num, valid_num), target_test, label='real')
plt.plot(np.linspace(1, valid_num, valid_num), L3_.T, label='predict')
plt.xlabel('# of Data')
plt.title('testing compare')
plt.legend(title='learning curve :')
plt.show()
















