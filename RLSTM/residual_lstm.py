import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import nltk
import copy
import statistics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
data_train = np.load('data/ACB.npy')
print(data_train.shape)
data_test = np.load('data/SLS.npy')
print(data_test.shape)
label_train = np.load('data/lbACB.npy')
label_test = np.load('data/lbSLS.npy')
print(label_train.shape)
print(label_test.shape)
print(type(data_train))
print(type(data_test))

num_layer = 2

class RLSTM(nn.Module):
  def __init__(self,input_size = 100 , hidden_size = 100, num_layer = num_layer):
    super(RLSTM,self).__init__()
    learn_models = []
    learn_models.append(nn.LSTM(input_size,hidden_size))
    self.hidden_size = hidden_size
    self.input_size = input_size
    # build iteration for following 
    for i in range(num_layer - 1):
      learn_models.append(nn.LSTM(hidden_size, hidden_size))
    self.lstm = nn.ModuleList(learn_models)
  def forward(self, inputs):
    # inputs : (batch_size, seq_len, num_feature)
    x = np.transpose(inputs,(1,0,2))  # (seq_len, batch_size, num_feature)
    lstm,_ = self.lstm[0](x)
    for i in range(num_layer -1):
      lstm_backup = copy.copy(lstm) 
      lstm,(h,c) = self.lstm[i+1](lstm)
      Wio = self.lstm[i+1].weight_ih_l0[3*self.hidden_size:].repeat(x.shape[1],1,1) # self.lstm2
      Wih = self.lstm[i+1].weight_hh_l0[3*self.hidden_size:].repeat(x.shape[1],1,1)
      bio = self.lstm[i+1].bias_ih_l0[3*self.hidden_size:].repeat(x.shape[1],1,1).view(-1,self.hidden_size,1)
      bih = self.lstm[i+1].bias_hh_l0[3*self.hidden_size:].repeat(x.shape[1],1,1).view(-1, self.hidden_size,1)
    h = h.view(-1,self.hidden_size,1)
    o_1 = torch.bmm(Wih,h) +  torch.bmm(Wio,lstm_backup[-1].view(-1,self.hidden_size,1)) + bio + bih
    o_1 = o_1.view(inputs.shape[0],self.hidden_size,1) ########### !!!!!!!!!!!!!
    h = h.view(inputs.shape[0],self.hidden_size,1)
    output = h + o_1*lstm_backup[-1].view(-1,self.hidden_size,1)

    return output

class Pred(nn.Module):
  def __init__(self, hidden_LSTM,hidden_size_1,hidden_size_2, input_dim, num_layer):
    super(Pred,self).__init__()
    self.hidden_LSTM = hidden_LSTM
    self.hidden_size_1 = hidden_size_1
    self.hidden_size_2 = hidden_size_2
    self.input_dim = input_dim
    self.lstm = RLSTM(input_dim, hidden_LSTM, num_layer)
    self.dense_1 = nn.Linear(hidden_LSTM,hidden_size_1) # 2  ->10
    self.dense_2 = nn.Linear(hidden_size_1,hidden_size_2)
    

  def forward(self,inputs):
    x = torch.FloatTensor(inputs)
    lstm = self.lstm(x).view(-1,self.hidden_LSTM) # chi lay cell cuoi cung de predict
    dense_1 = self.dense_1(lstm)
    output = self.dense_2(dense_1)
    return output

train_set  = data_train
y_train    = label_train
test_set   = data_test
y_test     = label_test

model  = Pred(100,100,1,6, num_layer)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)
loss = nn.MSELoss()

batch_size = 32
iter_all_batch = int(train_set.shape[0]/batch_size)
num_iters = 2000
step = 0
iter_index = 0
step_display = 10
step_test = 30
loss_average = torch.tensor(0.0)
accuracy_train = torch.tensor(0.0)
accuracy_test = torch.tensor(0.0)
train_results = []
test_results = []
loss_batch = []

def train_iteration(batch_size,step,model, optimizer,loss,train_set,train_label):
  batch = train_set[step*batch_size:(step+1)*batch_size]
  print('size of batch: ')
  print(batch.shape)
  groundtruth = torch.FloatTensor(train_label[step*batch_size:(step+1)*batch_size])
  optimizer.zero_grad()
  pred = model(batch)
  print('size of pred: ')
  print(pred.shape)
  loss = loss(pred,groundtruth)
  loss.backward()
  optimizer.step()
  
  return loss

def test_step(model, data, label_test):
  pred_test = model(data)
  label_test = torch.FloatTensor(label_test)
  error = nn.L1Loss()(pred_test,label_test)
  return error

best_model = copy.deepcopy(model)
min_error = 9999999999.0

for i in range(num_iters):
  if((step+1)*batch_size > train_set.shape[0]):
    step = 0
  cost= train_iteration(batch_size,step,model,optimizer,loss,train_set,y_train)
  train_results.append(cost)
  loss_batch.append(cost)
  print('step: ', iter_index, 'loss_per_batch: ', cost)
  
  if((iter_index+1)%step_test==0):
    error = test_step(model,test_set,y_test)
    print('step: ', iter_index, 'error_test: ', error)
    test_results.append(error)
    if(min_error > error):
      best_model = model
      min_error =error
  
  iter_index = iter_index + 1
  step = step +1
  
  print('------------------step: ', iter_index,' ------------------')

# Plot loss function and predict values
import matplotlib.pyplot as plt
plt.figure()
plt.plot(loss_batch, '-')
plt.legend('loss values')
plt.title('loss of training')
plt.show()

gt = y_test
prediction = best_model(test_set)
plt.figure()
plt.plot(gt, 'r--')
plt.plot(prediction.detach().numpy(), 'b--')
plt.legend(('groundtruth', 'predict'))
plt.title('prediction test set')
plt.show()

# Save model

torch.save(best_model.state_dict(),'modelRLSTM.h5')



# load model
new_model = Pred(100,100,1,6,num_layer) # chinh lai cac arguments
new_model.load_state_dict(torch.load('modelRLSTM.h5'))

# load params
for param_tensor in new_model.state_dict():
    print(param_tensor, "\t", new_model.state_dict()[param_tensor].size())

import json

#train_results = [[i.detach().numpy(), j.detach().numpy()] for (i,j) in train_results]
print('error testing results: ')
print(test_results)

# save data in json file
file = open('data_trained.json', 'w+')
np.save('train_results.npy',np.array(train_results))
np.save('test_results.npy',np.array(test_results))

train_load = np.load('train_results.npy', allow_pickle= True)


