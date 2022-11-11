import torch
import torch.nn as nn
import torch.utils.data as dset
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Hyper Parameters
# batch_size, epoch and iteration
LR = 0.001
#n_iters = 10000
#num_epochs = n_iters / (len(train) / batch_size)
#num_epochs = int(num_epochs)
num_epochs = 3
batch_size = 300

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判斷是否有GPU資源可用
#print(device)

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5,),(0.5,))])

# Data
train = datasets.MNIST(root='MNIST', download=True, train=True , transform=transform)
test = datasets.MNIST(root='MNIST', download=True, train=False , transform=transform)
# create validation set

train_imag = train.data / 255
train_target = train.targets
test_imag = test.data / 255
test_target = test.targets


train, valid = dset.random_split(train,[55000,5000])
train_loader = dset.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = dset.DataLoader(valid, batch_size=batch_size, shuffle=True)
test_loader = dset.DataLoader(test, batch_size=len(test.targets), shuffle=False)



#%%
'''
plt.figure()
plt.imshow(train_imag[3])
plt.colorbar()
plt.grid(False)
plt.title(train_target[3])
plt.show()
'''
#%%

# Model
class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(1,28,28)
        self.cov1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0) #output_shape=(64,24,24)
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(64,12,12)
        # Convolution 2
        self.cov2 = nn.Conv2d(in_channels=64, out_channels=16, kernel_size=7, stride=1, padding=0) #output_shape=(16,6,6)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,3,3)
        # Fully connected 1 ,#input_shape=(16*3*3)
        self.fc1 = nn.Linear(16 * 3 * 3, 128) 
        self.relu3 = nn.ReLU() # activation
        self.fc2 = nn.Linear(128, 64) 
        self.relu4 = nn.ReLU() # activation
        self.fc3 = nn.Linear(64, 10) 
 
    def forward(self, x):
        # Convolution 1
        cov1_out = self.cov1(x)
        relu1_out = self.relu1(cov1_out)
        # Max pool 1
        maxpool1_out = self.maxpool1(relu1_out)
        # Convolution 2 
        cov2_out = self.cov2(maxpool1_out)
        relu2_out = self.relu2(cov2_out)
        # Max pool 2 
        maxpool2_out = self.maxpool2(relu2_out)
        view_out = maxpool2_out.view(maxpool2_out.size(0), -1)
        fc1_out = self.fc1(view_out)
        relu3_out = self.relu3(fc1_out)
        fc2_out = self.fc2(relu3_out)
        relu4_out = self.relu4(fc2_out)
        fc3_out = self.fc3(relu4_out)
        return fc3_out , cov1_out ,maxpool1_out, cov2_out,maxpool2_out
    
#%%
model = CNN_Model()
print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR ,weight_decay= 1e-2 )   # optimize all cnn parameters
loss_func = nn.CrossEntropyLoss()   # the target label is not one-hotted
input_shape = (-1,1,28,28)

# Train
def fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, valid_loader, test_loader):
    # Traning the Model
    #history-like list for store loss & acc value
    training_loss = []
    training_accuracy = []
    validation_loss = []
    validation_accuracy = []
    testing_loss = []
    testing_accuracy = []
      
    for epoch in range(num_epochs):
        ########################################################
        #training model & store loss & acc / epoch
        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            train = Variable(images.view(input_shape))
            labels = Variable(labels)
            optimizer.zero_grad()
            outputs,_,_,_,_ = model(train)
            train_loss = loss_func(outputs, labels)
            train_loss.backward()
            optimizer.step()
            predicted = torch.max(outputs.data, 1)[1]
            total_train += len(labels)
            correct_train += (predicted == labels).float().sum()
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)
        training_loss.append(train_loss.data)
        ########################################################
        correct_valid = 0
        total_valid = 0
        for images, labels in valid_loader:
            test = Variable(images.view(input_shape))
            outputs,_,_,_,_ = model(test)
            val_loss = loss_func(outputs, labels)
            predicted = torch.max(outputs.data, 1)[1]
            total_valid += len(labels)
            correct_valid += (predicted == labels).float().sum()
        val_accuracy = 100 * correct_valid / float(total_valid)
        validation_accuracy.append(val_accuracy)
        validation_loss.append(val_loss.data)
        ####################################################
        correct_test = 0
        total_test = 0
        for images, labels in test_loader:
            test = Variable(images.view(input_shape))
            outputs,_,_,_,_ = model(test)
            test_loss = loss_func(outputs, labels)
            predicted = torch.max(outputs.data, 1)[1]
            total_test += len(labels)
            correct_test += (predicted == labels).float().sum()
        test_accuracy = 100 * correct_test / float(total_test)
        testing_accuracy.append(test_accuracy)
        testing_loss.append(test_loss.data)
        ####################################################
        
        
        
        print('Train Epoch: {}/{} \n Tran_Loss: {:.3f} Tran_acc: {:.3f}% \n Vali_Loss: {:.3f} Vali_accuracy: {:.3f}% \n Test_Loss: {:.3f} Test_accuracy: {:.3f}%'.format(epoch+1, num_epochs, train_loss.data, train_accuracy, val_loss.data, val_accuracy,  test_loss.data,  test_accuracy))
    
    return training_loss, training_accuracy, validation_loss, validation_accuracy,  testing_loss,  testing_accuracy

training_loss, training_accuracy, validation_loss, validation_accuracy , testing_loss, testing_accuracy= fit_model(model, loss_func, optimizer, input_shape, num_epochs, train_loader, valid_loader,test_loader)

'''
test_l = dset.DataLoader(test, batch_size=len(test.targets), shuffle=False)
#images_draw = enumerate(test_l)
test_draw= Variable(test.data.view(input_shape))
predict = model(test_draw)
'''
#%% 折線圖
# visualization
plt.plot(torch.linspace(1,len(training_loss),len(training_loss),dtype=int), training_loss,'b-', label='Training_loss')
plt.plot(torch.linspace(1,len(validation_loss),len(validation_loss),dtype=int), validation_loss,'g-', label='validation_loss')
plt.plot(torch.linspace(1,len(testing_loss),len(testing_loss),dtype=int), testing_loss,'r-', label='Test_loss')
plt.title('Training & Validation loss')
plt.xlabel('Number of epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(torch.linspace(1,len(training_accuracy),len(training_accuracy),dtype=int), training_accuracy, 'b-', label='Training_accuracy')
plt.plot(torch.linspace(1,len(validation_accuracy),len(validation_accuracy),dtype=int), validation_accuracy, 'g-', label='Validation_accuracy')
plt.plot(torch.linspace(1,len(testing_accuracy),len(testing_accuracy),dtype=int), testing_accuracy, 'r-', label='Test_accuracy')
plt.title('Training & Validation accuracy')
plt.xlabel('Number of epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

#%% 柱狀圖
# Print model's state_dict
print("Model's state_dict:")
A=['cov1.weight' , 'cov2.weight' , 'fc3.weight']
for param_tensor in model.state_dict():
    #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
    if param_tensor in A:
        #print(param_tensor, "t/" ,model.state_dict()[param_tensor])
        hist = model.state_dict()[param_tensor].numpy()
        hist = hist.reshape(-1,1)
        plt.figure()
        plt.xlabel('Value')
        plt.ylabel('Number')
        plt.title(param_tensor)
        plt.hist(hist,bins=50)
        plt.show()
        
#%% 1-2

for images, labels in test_loader:
    test = Variable(images.view(input_shape))
    outputs, cov1_out , maxpool1_out, cov2_out, maxpool2_out = model(test)
    test_loss = loss_func(outputs, labels)
    predicted = torch.max(outputs.data, 1)[1]
    pred_ = predicted.tolist()
    label_ = labels.tolist()
correct_index = []
incorrect_index = []
for k in range(len(label_)):
    if label_[k] == pred_[k]:
        correct_index.append(k)
    elif  label_[k] != pred_[k]:
        incorrect_index.append(k)

plt.figure()
plt.imshow(test_imag[correct_index[0]])
plt.colorbar()
plt.grid(False)
plt.title('predict = %d label = %d'%(pred_[correct_index[0]],label_[correct_index[0]]))
plt.show()


plt.figure()
plt.imshow(test_imag[incorrect_index[0]])
plt.colorbar()
plt.grid(False)
plt.title('predict = %d label = %d'%(pred_[incorrect_index[0]],label_[incorrect_index[0]]))
plt.show()
        
#%%
show_fig_num = 65

plt.figure()
plt.imshow(test_imag[show_fig_num])
plt.colorbar()
plt.grid(False)
plt.show()

plt.figure(figsize=(8,8))
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cov1_out[show_fig_num][i].detach().numpy())
plt.show()

plt.figure(figsize=(8,8))
for i in range(64):
    plt.subplot(8,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(maxpool1_out[show_fig_num][i].detach().numpy())
plt.show()

plt.figure(figsize=(4,4))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(cov2_out[show_fig_num][i].detach().numpy())
plt.show()

        
plt.figure(figsize=(4,4))
for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(maxpool2_out[show_fig_num][i].detach().numpy())
plt.show()       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        