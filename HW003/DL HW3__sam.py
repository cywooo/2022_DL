#!/usr/bin/env python
# coding: utf-8

# REF:
#     https://www.itread01.com/content/1543433296.html

# In[1]:


import tensorflow as tf
import numpy as np
import os
import time
import tensorflow
import matplotlib.pyplot as plt


# In[2]:


tf.__version__


# In[3]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# # Load data

# In[4]:


#2. 資料預處理
with open('shakespeare_train.txt') as f:
    # text 是個字串
    text = f.read()
#2. 資料預處理
with open('shakespeare_valid.txt') as f:
    # text 是個字串
    text_valid = f.read()
#print(text_valid[:1000])


# In[5]:


# 3. 將組成文字的字元全部提取出來，注意 vocab是個list
vocab = sorted(set(text))
vocab_valid = sorted(set(text_valid))


# In[6]:


print('Length of text: {} characters'.format(len(text)))
print('{} unique characters'.format(len(vocab)))

print('Length of text_valid: {} characters'.format(len(text_valid)))
print('{} unique characters_valid'.format(len(vocab_valid)))


# In[7]:


# 4. 建立text-->int的對映關係
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)
text_as_int = np.array([char2idx[c] for c in text])

char2idx_valid = {u:i for i, u in enumerate(vocab_valid)}
idx2char_valid = np.array(vocab_valid)
text_as_int_valid = np.array([char2idx_valid[c] for c in text_valid])


# In[8]:


# 5. 借用dataset的batch方法，將text劃分為定長的句子
seq_length = 200
examples_per_epoch = len(text)//seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
'''for i in char_dataset.take(5):
    print(idx2char[i.numpy()])'''

    
# 5. 借用dataset的batch方法，將text劃分為定長的句子
examples_per_epoch_valid = len(text_valid)//seq_length
char_dataset_valid = tf.data.Dataset.from_tensor_slices(text_as_int_valid)

for i in char_dataset_valid.take(5):
    print(idx2char_valid[i.numpy()])


# In[9]:


# 這裡batch_size 加1的原因在於，下面對inputs和labels的生成。labels比inputs多一個字元
sequences = char_dataset.batch(seq_length+1, drop_remainder=True)
'''for item in sequences.take(5):
    print(repr(''.join(idx2char[item.numpy()])))'''
    
sequences_valid = char_dataset_valid.batch(seq_length+1, drop_remainder=True)
for item in sequences_valid.take(5):
    print(repr(''.join(idx2char_valid[item.numpy()])))


# In[10]:


# 6. 將每個句子劃分為inputs和labels。例如：hello,inputs = hell,label=ello
def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)
dataset_valid = sequences_valid.map(split_input_target)


# In[11]:


for input_example, target_example in  dataset.take(1):
    print('Input data: ', repr(''.join(idx2char[input_example.numpy()])))
    print('Target data:', repr(''.join(idx2char[target_example.numpy()])))
    
for input_example_valid, target_example_valid in  dataset_valid.take(1):
    print('Input data: ', repr(''.join(idx2char_valid[input_example_valid.numpy()])))
    print('Target data:', repr(''.join(idx2char_valid[target_example_valid.numpy()])))


# In[12]:


for i, (input_idx, target_idx) in enumerate(zip(input_example[:5], target_example[:5])):
    print("Step {:4d}".format(i))
    print("  input: {} ({:s})".format(input_idx, repr(idx2char[input_idx])))
    print("  expected output: {} ({:s})".format(target_idx, repr(idx2char[target_idx])))
    
for i_valid, (input_idx_valid, target_idx_valid) in enumerate(zip(input_example_valid[:5], target_example_valid[:5])):
    print("Step {:4d}".format(i_valid))
    print("  input: {} ({:s})".format(input_idx_valid, repr(idx2char_valid[input_idx_valid])))
    print("  expected output: {} ({:s})".format(target_idx_valid, repr(idx2char_valid[target_idx_valid])))    


# In[13]:


# 7. 將句子劃分為一個個batch
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE
BUFFER_SIZE = 10000
# drop_remainder 一般需要設定為true，表示當最後一組資料不夠劃分為一個batch時，將這組資料丟棄
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)



BATCH_SIZE = 64
steps_per_epoch_valid = examples_per_epoch_valid//BATCH_SIZE
BUFFER_SIZE = 10000
dataset_valid = dataset_valid.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# In[14]:


# 8. 模型搭建
# Length of the vocabulary in chars
vocab_size = len(vocab)
# The embedding dimension
embedding_dim = 256
# Number of RNN units
rnn_units = 1024
model = tf.keras.Sequential()
# 這裡是字元embedding，所以是字符集大小*embedding_dim
model.add(tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,
                                    batch_input_shape=[BATCH_SIZE,None])),


model.add(tf.keras.layers.LSTM(units=rnn_units,
                              return_sequences=True,
                              recurrent_initializer='glorot_uniform',
                               dropout=0.2,
                                  stateful=True)),
                                                           
model.add(tf.keras.layers.Dense(units=vocab_size))


# In[15]:


for input_example_batch, target_example_batch in dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    
for input_example_batch_valid, target_example_batch_valid in dataset_valid.take(1):
    example_batch_predictions_valid = model(input_example_batch_valid)
    print(example_batch_predictions_valid.shape, "# (batch_size, sequence_length, vocab_size)")


# In[16]:


model.summary()


# In[17]:


sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()


# In[18]:


sampled_indices


# In[19]:


print("Input: \n", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Next Char Predictions: \n", repr("".join(idx2char[sampled_indices ])))


# In[20]:


def loss(labels, logits):
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

example_batch_loss = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)")
print("scalar_loss:      ", example_batch_loss.numpy().mean())


# In[21]:


#tf.enable_eager_execution()

# 9. 模型配置
# optimizer 必須為 tf.train 下的opt，不能是keras下的opt
#model.compile(optimizer=tf.train.AdamOptimizer(),loss=tf.losses.sparse_softmax_cross_entropy)
    #some error REF: 
    #https://stackoverflow.com/questions/53272808/tensorflow-keras-typeerror-value-passed-to-parameter-labels-has-datatype-f
#model.compile(optimizer='adam', loss=loss)
model.compile(optimizer='adam', loss=loss)


# In[26]:


# 10 .設定回撥函式
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")
checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./logs/')
#REF:https://github.com/ibab/tensorflow-wavenet/issues/255

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='logs')


# In[23]:


#model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))


# In[ ]:


# 11. 訓練模型,repeat() 表示dataset無限迴圈，不然這裡資料可能不夠30個epochs
history = model.fit(dataset.repeat(),epochs=10,validation_data=dataset_valid,
          steps_per_epoch=steps_per_epoch,
          callbacks=[checkpoint_callback,tensorboard_callback])


# In[ ]:


# 12 .模型儲存
# 儲存為keras模型格式
model.save_weights(filepath='./models/gen_text_with_char_on_rnn.h5',save_format='h5')
# 儲存為TensorFlow的格式
model.save_weights(filepath='./models/gen_text_with_char_on_rnn_check_point')


# In[ ]:


history_dict = history.history
print(history_dict.keys())


# In[ ]:


fig = plt.figure()
plt.subplot(1,1,1)
plt.plot(history.history['val_loss'])
#plt.plot(history.history['val_acc'])

plt.title('Validation error')
plt.ylabel('Validation loss')
plt.xlabel('Iteration')
plt.legend([ 'Validation Accuracy'], loc='lower right')
#rcParams['figure.figsize'] = 5, 3


# In[ ]:


plt.subplot(2,1,2)
plt.plot(history.history['loss'])
plt.title('Learning Curve')
plt.ylabel('Loss')
plt.xlabel('Iteration')
plt.legend(['Cross entropy'], loc='upper right')

plt.tight_layout()
#rcParams['figure.figsize'] = 5, 4


# In[ ]:


tf.train.latest_checkpoint(checkpoint_dir)


# In[ ]:


BATCH_SIZE=1
model = tf.keras.Sequential()
# 這裡是字元embedding，所以是字符集大小*embedding_dim
model.add(tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=embedding_dim,
                                    batch_input_shape=[BATCH_SIZE,None]))
'''model.add(tf.keras.layers.GRU(units=rnn_units,
                              return_sequences=True,
                              recurrent_initializer='glorot_uniform',
                                  stateful=True))
'''
model.add(tf.keras.layers.SimpleRNN(units=rnn_units,
                              return_sequences=True,
                              recurrent_initializer='glorot_uniform',
                                  stateful=True)),                            
model.add(tf.keras.layers.Dense(units=vocab_size))


# In[ ]:


model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


# In[ ]:


model.summary()


# In[ ]:


# 13. 模型生成文字
def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # You can change the start string to experiment
    start_string = 'ROMEO'

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


# In[ ]:


print(generate_text(model))


# In[ ]:


print(generate_text(model, start_string="ROMEO: "))


# ## Customized Training with hidden set reset per epoch

# In[28]:


trloss_ep10=[]


# In[29]:



seq_length = 100#chaange different sequence length to check
examples_per_epoch = len(text)//seq_length
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)


# In[30]:


sequences = char_dataset.batch(seq_length+1, drop_remainder=True)


# In[31]:


def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text
dataset = sequences.map(split_input_target)
dataset_valid = sequences_valid.map(split_input_target)


# In[32]:


BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE
BUFFER_SIZE = 10000
# drop_remainder 一般需要設定為true，表示當最後一組資料不夠劃分為一個batch時，將這組資料丟棄
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)


# In[62]:


# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units =128#try with different hidden state size


# In[63]:


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                  batch_input_shape=[batch_size, None]),
        tf.keras.layers.LSTM(units=rnn_units,
                              return_sequences=True,
                              recurrent_initializer='glorot_uniform',
                               dropout=0.2,
                                  stateful=True),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model


# In[64]:


model = build_model(
    vocab_size=len(vocab),
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=BATCH_SIZE)


# In[65]:


model.summary()


# In[66]:


optimizer = tf.keras.optimizers.Adam()


# In[67]:


@tf.function
def train_step(inp, target):
    with tf.GradientTape() as tape:
        predictions = model(inp)
        loss = tf.reduce_mean(
            tf.keras.losses.sparse_categorical_crossentropy(
                target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss


# In[68]:


# Training step
EPOCHS = 10

seq_try=[20,50,100,150,200,300 ]
for epoch in range(EPOCHS):
    start = time.time()

    # resetting the hidden state at the start of every epoch
    model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)

        if batch_n % 100 == 0:
            template = 'Epoch {} Batch {} Loss {}'
            print(template.format(epoch + 1, batch_n, loss))
        
    # saving (checkpoint) the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))
    
    print('Epoch {} Loss {:.4f}'.format(epoch + 1, loss))
    print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))


# In[40]:


trloss_ep10_simplernn=[1.518121361732483,1.4043,1.3355,1.3488 ,1.3897 ,
             1.3809]
seq_try_simplernn=[20,50,100,150,200,300 ] 

trloss_ep10=[1.3929,1.2524,1.1730,1.1843,1.1683,1.1896]
seq_try=[20,50,100,150,200,300 ] 


# In[ ]:


fig = plt.figure()
plt.subplot(1,1,1)
plt.plot(seq_try,trloss_ep10)
#plt.plot(history.history['val_acc'])

plt.title('Training loss with different seq_length LSTM')
plt.ylabel('Training Loss')
plt.xlabel('Sequence Length')
plt.legend(['Training Loss', 'Sequence Length'], loc='lower right')


# In[69]:


hidden_state_size_lstm=[128,256,512,1024,1536,2048]
trloss_ep10_lstm_hs=[1.5210,1.3625,1.2790,1.1730,1.1216,1.0177]


# In[70]:


fig = plt.figure()
plt.subplot(1,1,1)
plt.plot(hidden_state_size_lstm,trloss_ep10_lstm_hs)
#plt.plot(history.history['val_acc'])

plt.title('Training loss with dif hidden state size LSTM')
plt.ylabel('Training Loss')
plt.xlabel('Hidden state size')
plt.legend(['Training Loss', 'Sequence Length'], loc='lower right')


# In[74]:


Hidden_state_size_simpleRNN=[128,256,512,1024,1536,2048]
Training_loss_epoch10_simple_RNN=[1.6008,1.5061,1.3996,1.3355,1.3301,1.4338]


# In[75]:


fig = plt.figure()
plt.subplot(1,1,1)
plt.plot(Hidden_state_size_simpleRNN,Training_loss_epoch10_simple_RNN)
#plt.plot(history.history['val_acc'])

plt.title('Training loss with dif hidden state size Simple RNN')
plt.ylabel('Training Loss')
plt.xlabel('Hidden state size')
plt.legend(['Training Loss', 'Sequence Length'], loc='lower right')


# In[ ]:


model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))


# In[ ]:


# 13. 模型生成文字
def generate_text(model, start_string):
    # Evaluation step (generating text using the learned model)

    # Number of characters to generate
    num_generate = 1000

    # You can change the start string to experiment
    start_string = 'ROMEO'

    # Converting our start string to numbers (vectorizing)
    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    # Empty string to store our results
    text_generated = []

    # Low temperatures results in more predictable text.
    # Higher temperatures results in more surprising text.
    # Experiment to find the best setting.
    temperature = 1.0

    # Here batch size == 1
    model.reset_states()
    for i in range(num_generate):
        predictions = model(input_eval)
        # remove the batch dimension
        predictions = tf.squeeze(predictions, 0)

        # using a multinomial distribution to predict the word returned by the model
        predictions = predictions / temperature
        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

        # We pass the predicted word as the next input to the model
        # along with the previous hidden state
        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))


# In[ ]:


print(generate_text(model, start_string="ROMEO: "))

