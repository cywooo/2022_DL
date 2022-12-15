import unidecode
import string
import random
import torch
#%%Prepare data
all_characters = string.printable
n_characters = len(all_characters)

with open('shakespeare_train.txt') as f:
    file = f.read()
    
with open('shakespeare_valid.txt') as f:
    text_valid = f.read()
    
file_len = len(file)

#%%
chunk_len = 1000

def random_chunk():
    start_index = random.randint(0, file_len - chunk_len)
    end_index = start_index + chunk_len + 1
    return file[start_index:end_index]

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
        self.rnn = nn.RNN(hidden_size, hidden_size, n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)
    
    def forward(self, input, hidden):
        input = self.encoder(input.view(1, -1))
        output, hidden = self.rnn(input.view(1, 1, -1), hidden)
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

print(char_tensor('abcDEF'))

def random_training_set():    
    chunk = random_chunk()
    inp = char_tensor(chunk[:-1])
    target = char_tensor(chunk[1:]) ##向後一位的預測
    return inp, target

#%% Evaluating
def evaluate(prime_str='A', predict_len=100, temperature=0.8):
    hidden = decoder.init_hidden()
    prime_input = char_tensor(prime_str)
    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[p], hidden)
    inp = prime_input[-1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]
        
        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = char_tensor(predicted_char)

    return predicted
#%% Training
import time, math

def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def train(inp, target):
    hidden = decoder.init_hidden()
    decoder.zero_grad()
    loss = 0

    for c in range(chunk_len):
        output, hidden = decoder(inp[c], hidden)
        #print(output.shape)
        #print(target[c].view(1).shape)
        loss += criterion(output, target[c].view(-1))####小心

    loss.backward()
    decoder_optimizer.step()
    ##return loss.data[0] / chunk_len
    return loss.item() / chunk_len


n_epochs = 300
print_every = 100
plot_every = 10
hidden_size = 256
n_layers = 2
lr = 0.005

decoder = RNN_(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

start = time.time()
all_losses = []
loss_avg = 0

for epoch in range(1, n_epochs + 1):
    inp, target = random_training_set()
    loss = train(inp, target)       
    loss_avg += loss

    if epoch % print_every == 0:
        print('\n','[%s (%d %d%%) %.4f]' % (time_since(start), epoch, epoch / n_epochs * 100, loss))
        print(evaluate('Wh', 100), '\n')

    if epoch % plot_every == 0:
        all_losses.append(loss_avg / plot_every)
        loss_avg = 0
        
#%%Plotting the Training Losses
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# %matplotlib inline

plt.figure()
plt.plot(all_losses)
print('\n','!!! Outcome temperature=0.8 !!!','\n')
print(evaluate('Th', 300, temperature=0.8))

'''
print('\n','!!! Outcome temperature=0.2 !!!','\n')
print(evaluate('Th', 200, temperature=0.2))

print('\n','!!! Outcome temperature=1.4 !!!','\n')
print(evaluate('Th', 200, temperature=1.4))
'''
