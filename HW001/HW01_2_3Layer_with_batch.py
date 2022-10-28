import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

file_path = os.getcwd()+'\ionosphere_data.csv' #使CSV檔的位置隨著資料夾變動
Raw_data_DF = pd.read_csv(file_path,header=None) #利用PANDAS將檔案的資料吃進來
Raw_data_DF =Raw_data_DF.sample(frac=1,replace=False ,random_state=np.random.randint(1e6),axis=0) #打亂排序

def pre_trea(Raw_data_DF): 
    target_OH = pd.get_dummies(Raw_data_DF[34]) #pandas分離出 Orientation 做one hot
    Raw_data_DF=Raw_data_DF.drop(34,axis=1) #刪除舊資料
    Raw_data_DF = pd.concat([Raw_data_DF,target_OH],axis=1) #嫁接在一起
    data_train_DF =Raw_data_DF.sample(frac=0.8,replace=False ,random_state=np.random.randint(1e6),axis=0) #training&testing分群 取80%
    data_test_DF =Raw_data_DF[~Raw_data_DF.index.isin(data_train_DF.index)] #剩下的25%
    target_train_DF= pd.concat([data_train_DF.pop('b'),data_train_DF.pop('g')],axis=1)
    target_test_DF= pd.concat([data_test_DF.pop('b'),data_test_DF.pop('g')],axis=1) #Bad在前 CLA1 Good在後 CLA2
    data_train=data_train_DF.values         #將資料轉為NP
    data_test=data_test_DF.values           #將資料轉為NP
    target_train=target_train_DF.values     #將資料轉為NP
    target_test=target_test_DF.values       #將資料轉為NP

    return data_train,data_test,target_train,target_test

#定義ReLU function
def ReLU(x):
    x[x[:,:]<0]=0
    return x

#定義deReLU function
def dReLU(x):
    x[x[:,:]<0]=0
    x[x[:,:]>=0]=1
    return x

#定義Softmax function
def Softmax(x):
    return (np.exp(x)/np.sum(np.exp(x),axis = 0 ,keepdims=True))

def dsoft(x):
    return np.exp(x)/np.sum(np.exp(x),axis = 0,keepdims=True)-np.exp(2*x)*np.power(np.sum(np.exp(x),axis = 0,keepdims=True),-2)

#active function
def active(x):
    return x

#active function differential
def dactive(x):
    y,x =x.shape
    out= np.ones((y,x))
    return out

def accuracy(w1, w2, w3, b1, b2, b3, target, features):
    l1, l2, l3, a1, a2, a3 =foward(features, w1, w2, w3, b1, b2, b3)
    l3=l3/np.amax(l3 ,axis=0,keepdims=True)
    l3[l3[:,:]!=1]=0
    return np.sum(l3.T*target)/len(target)
    
def cost ( w1, w2, w3, b1, b2, b3, target, features):
    l1, l2, l3, a1, a2, a3 =foward(features, w1, w2, w3, b1, b2, b3)
    return np.average(-1*target*(np.log(l3.T)))

def scatter_draw(a_out,y,n):
    y=y/np.amax(y ,axis=0,keepdims=True)
    a_out=np.concatenate((a_out,y), axis=0)
    class_1=a_out[:,a_out[2,:]==1]
    class_2=a_out[:,a_out[2,:]!=1]
    plt.scatter(class_1[0,:],class_1[1,:],marker='o',label='Bad')
    plt.scatter(class_2[0,:],class_2[1,:],marker='+',label='Good')
    plt.legend()
    plt.title('Scatter chart epoch =%d'%(n))
    plt.show()  

#參數設定
np.random.seed(54321) #54321
learning_rate = 1e-5    #1e-5   #1e-5
epoch = 5000           #5000   #15000
batch = 100             #100     #281

Num_of_hiden1 = 128  #128
Num_of_hiden2 = 64  #64
Num_of_outLayer = 2 
draw_scatter=[10,20,50,100,500,2000,4000,epoch]

features_train,features_test,target_train,target_test = pre_trea(Raw_data_DF) #CALL資料預處裡函式

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
accuracy_train=np.zeros(epoch)
accuracy_test=np.zeros(epoch)

#feed forward
def foward(x, w1, w2, w3, b1, b2, b3):
    a1 = np.dot(w1.T, x.T) + b1
    l1 = ReLU(a1)
    a2 = np.dot(w2.T, l1) + b2
    l2 = ReLU(a2)
    a3 = np.dot(w3.T, l2) + b3
    l3 = Softmax(a3)
    return l1, l2, l3, a1, a2, a3

#def back(x, n, w1, w2, w3, b1, b2, b3):
def back( x, t, w1, w2, w3, b1, b2, b3, l1, l2, l3, a1, a2, a3, rl, n): 
    Edy=((-1*t.T/l3)+((1-t.T)/(1-l3)))
    #Edy=l3-t.T #
    dw3 =np.dot( l2,  (Edy*dsoft(a3)).T)
    dw2 =np.dot( l1,  (np.dot( w3, (Edy*dsoft(a3)) )*dReLU(a2)).T )    
    dw1 =np.dot( x.T, (np.dot( w2, ((np.dot( w3, (Edy*dsoft(a3)) )*dReLU(a2))) )*dReLU(a1)).T)
    db3 =Edy*dsoft(a3)
    db2 =np.dot( w3, Edy*dsoft(a3))*dReLU(a2)
    db1 =np.dot( w2,np.dot( w3, Edy*dsoft(a3))*dReLU(a2))*dReLU(a1)
    w1= w1-rl*dw1
    w2= w2-rl*dw2
    w3= w3-rl*dw3
    b1= b1-rl*np.sum(db1, axis = 1,keepdims=True)/batch
    b2= b2-rl*np.sum(db2, axis = 1,keepdims=True)/batch
    b3= b3-rl*np.sum(db3, axis = 1,keepdims=True)/batch
    return w1, w2, w3, b1, b2, b3
    
#pick arrays as batch
def batch_pick( n, train_num, counter):
    features_train_batch = np.zeros((n, num_of_features))
    target_train_batch = np.zeros((n, num_of_targets))
    if((counter+n)>(train_num-1)):
        return features_train[counter:][:] , target_train[counter:][:]
    features_train_batch = features_train[counter:counter+n][:]
    target_train_batch = target_train[counter:counter+n][:]
    return features_train_batch, target_train_batch
 
# main
for i in range(epoch):
    for j in range(batch_epoch+1):   
        features_train_batch,target_train_batch = batch_pick(batch ,train_num , j)
        L1, L2, L3, A1, A2, A3 = foward(features_train_batch, w1, w2, w3, b1, b2, b3)
        w1, w2, w3, b1, b2, b3 = back(features_train_batch,target_train_batch,w1, w2, w3, b1, b2, b3, L1, L2, L3, A1, A2, A3, learning_rate, batch)
    if i+1 in draw_scatter:
        _, _, l3_, _, _, a3_ = foward(features_train, w1, w2, w3, b1, b2, b3)
        scatter_draw(a3_,l3_,i+1)
    #if i in LR_C:
        #learning_rate=learning_rate/10
    accuracy_train[i] = accuracy( w1, w2, w3, b1, b2, b3 , target_train, features_train)
    accuracy_test[i] = accuracy( w1, w2, w3, b1, b2, b3 , target_test, features_test)
    tloss_draw[i] = cost( w1, w2, w3, b1, b2, b3 , target_train, features_train)
    vloss_draw[i] = cost( w1, w2, w3, b1, b2, b3 , target_test, features_test)
    print('Training parameters: epochs = %d tloss = %f vloss = %f' % (i+1, tloss_draw[i], vloss_draw[i]))
  
plt.plot(np.linspace(1, epoch, epoch), tloss_draw, label=' Training')
plt.plot(np.linspace(1, epoch, epoch), vloss_draw, label=' Testting',color='red', linestyle='dashed')
plt.xlabel('Epoch with Network [%d-%d-%d-%d]'%(num_of_features,Num_of_hiden1,Num_of_hiden2,Num_of_outLayer))
plt.ylabel('Loss')
plt.title('Learning curve chart \n b= %d  epoch= %d Lr= %.1e\n Final training loss %.3f ,testing loss %.3f'%(batch,epoch,learning_rate,tloss_draw[-1], vloss_draw[-1]))
plt.legend(title='Learning curve :')
plt.show()

plt.plot(np.linspace(1, epoch, epoch), accuracy_train, label='training accuracy')
plt.plot(np.linspace(1, epoch, epoch), accuracy_test, label='testning accuracy')
plt.legend()
plt.xlabel('Epoch with Network [%d-%d-%d-%d]'%(num_of_features,Num_of_hiden1,Num_of_hiden2,Num_of_outLayer))
plt.ylabel('Accuracy')
plt.title('Accuracy Chart \n b= %d  epoch= %d Lr= %.1e\n Final training accuracy %.3f & testing accuracy %.3f'%(batch,epoch,learning_rate,accuracy_train[-1],accuracy_test[-1]))
plt.legend()
plt.show()

plt.plot(np.linspace(1, epoch, epoch), 1-accuracy_train, label='training error')
plt.plot(np.linspace(1, epoch, epoch), 1-accuracy_test, label='testning error')
plt.legend()
plt.xlabel('Epoch with Network [%d-%d-%d-%d]'%(num_of_features,Num_of_hiden1,Num_of_hiden2,Num_of_outLayer))
plt.ylabel('Error rate')
plt.title('Error rate Chart \n b= %d  epoch= %d Lr= %.1e\n Final training error %.3f & testing error %.3f'%(batch,epoch,learning_rate,1-accuracy_train[-1],1-accuracy_test[-1]))
plt.legend()
plt.show()

















