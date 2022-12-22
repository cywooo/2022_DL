import unidecode
import string
import random
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
#%%Prepare data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # 判斷是否有GPU資源可用
print(device)

all_characters = string.printable
n_characters = len(all_characters)

with open('shakespeare_train.txt') as f:
    text_train = f.read()
    
with open('shakespeare_valid.txt') as f:
    text_valid = f.read()
    
train_len = len(text_train)
valid_len = len(text_valid)
#%%
chunk_len = 200
def random_chunk():
    start_index = random.randint(0, train_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return text_train[start_index:end_index]

valid_chunk_len = chunk_len
def valid_chunk():
    start_index = random.randint(0, valid_len - valid_chunk_len)
    end_index = start_index + valid_chunk_len + 1
    return text_valid[start_index:end_index]

#%% Build the Model
import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN_(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1):
        super(RNN_, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.encoder = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input_ = self.encoder(input.view(1, -1))
        h0 = torch.zeros(self.n_layers, input_.size(1), self.hidden_size).to(device) 
        c0 = torch.zeros(self.n_layers, input_.size(1), self.hidden_size).to(device)
        '''
        print(input.size())
        print(input_.size(2))
        print(input_.size())
        print(h0.size())
        '''
        
        output, hidden = self.lstm(input_.view(1,1,-1),(h0, c0))
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
    
    
#%% Inputs and Targets     OKOKOKOK   
# Turn string into list of longs
def char_tensor(string):
    tensor = torch.zeros(len(string)).long()
    for c in range(len(string)):
        tensor[c] = all_characters.index(string[c])
    return Variable(tensor)

def random_training_set():    
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:]) ##向後一位的預測
    return inp, target

def random_validing_set():
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:]) ##向後一位的預測
    return inp, target   


#%% Evaluating
def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden().cuda()
    prime_input = char_tensor(prime_str)
    prime_input = prime_input.cuda()
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
        
    inp = prime_input[-1] 
    
    for p in range(predict_len):
        #hidden = hidden.cuda()
        '''
        print(p)
        print(inp.type())
        print(hidden.type())
        '''
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)
        inp = inp.cuda()

    return predicted
#%%get V loss
def get_V_loss(inp, target):
    hidden = decoder.init_hidden().cuda()
    decoder.zero_grad()
    loss = 0
    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].view(-1))

    return loss.item() / valid_chunk_len

#%% Training
import time, math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(inp, target):
    inp = inp.cuda()
    hidden = decoder.init_hidden().cuda()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        #print(inp[c])
        output, hidden = decoder(inp[c], hidden)
        loss += criterion(output, target[c].view(-1))
    
    loss.backward()
    decoder_optimizer.step()
    return loss.item() / chunk_len


n_epochs = 500
print_every = 100
plot_every = 10
hidden_size = 512
n_layers = 1
lr = 0.005

decoder = RNN_(n_characters, hidden_size, n_characters, n_layers).cuda()
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
V_losses = []
train_loss_avg = 0

for epoch in range(1, n_epochs + 1):
    inp, target = random_training_set()
    inp = inp.cuda()
    target = target.cuda()
    train_loss = train(inp, target)  
    train_loss_avg += train_loss

    if epoch % print_every == 0: ##過度期的輸出
        print('\n','[%s (epoch:%d %d%%) tloss: %.4f ]' % (time_since(start), epoch, epoch / n_epochs * 100, train_loss))
        print(evaluate('Wh', 200), '\n')

    if epoch % plot_every == 0:
        V_inp, V_target = random_validing_set()
        V_inp = V_inp.cuda()
        V_target = V_target.cuda()
        V_loss = get_V_loss(V_inp, V_target)  
        V_losses.append(V_loss)   
        all_losses.append(train_loss_avg / plot_every)
        train_loss_avg = 0
        
#%%Plotting the Training Losses
import matplotlib.pyplot as plt
# %matplotlib inline

print(decoder)

plt.plot(all_losses, label='training Loss')
plt.plot(V_losses, label='validation Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Learning Curve')
plt.legend()
plt.show()

print('\n','[Final tloss: %.4f Final vloss: %.4f ]' % (all_losses[-1],V_losses[-1]))

print('\n','!!! Outcome temperature=0.8 !!!','\n')
print(evaluate('JULIET:', 300, temperature=0.8))

'''
print('\n','!!! Outcome temperature=0.2 !!!','\n')
print(evaluate('ROMEO:', 200, temperature=0.2))

print('\n','!!! Outcome temperature=1.4 !!!','\n')
print(evaluate('ROMEO:', 200, temperature=1.4))
'''
