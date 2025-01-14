
import torch
import torch.nn as nn
import numpy as np
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import nltk
import copy
import statistics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from torch.autograd import Variable
from model import Pred
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)


def numpy_to_tvar(x):
  # required grad is False if not declare
  return Variable(torch.from_numpy(x).type(torch.FloatTensor)).to(device)


data = pd.read_csv('~/DARNN/data/nasdaq100_padding.csv')
stocks_name = list(data.columns)
# Preprocessing
scale = StandardScaler().fit(data)
proc_dat = scale.transform(data)

print(proc_dat.shape)

steps = 10
name_stock = "NDX"
databatch = [] # luu cac batch , size = (num of batch,steps,6)
labels = []

index_stock = stocks_name.index(name_stock)
target = proc_dat[:,index_stock]
arr = np.array(proc_dat)

for i in range(len(arr)-steps):
    databatch.append(arr[i:i+steps])
    labels.append(target[i+steps])
# save file numpy .npy
np.save('nasdaq.npy',databatch)
np.save('lb_nasdaq.npy',labels)

# load file numpy .npy

data = np.load('nasdaq.npy')
labels = np.load('lb_nasdaq.npy')
print(np.shape(data))
print(np.shape(labels))

num_layer = 2

'''class RLSTM(nn.Module):
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
    h = lstm[-2].view(-1,self.hidden_size,1)
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
    self.batch = batch_size
    self.lstm = RLSTM(input_dim, hidden_LSTM, num_layer)
    self.dense_1 = nn.Linear(hidden_LSTM,hidden_size_1) # 2  ->10
    self.dense_2 = nn.Linear(hidden_size_1,hidden_size_2)
    self.bias = Variable(
          torch.empty((hidden_size_2,)).normal_(mean=0,std=1), 
                       requires_grad=True)
  def forward(self,inputs):
    x = torch.FloatTensor(inputs)
    lstm = self.lstm(x).view(-1,self.hidden_LSTM)# chi lay cell cuoi cung de predict
    dense_1 = self.dense_1(lstm) 
    output = self.dense_2(dense_1) + self.bias
    return output
'''

num_data = np.shape(data)[0]
num_train =  35090#int(num_data*0.65)
num_valid = 2730 #int(num_data*0.15)
num_test = 2730 #num_data - num_train - num_valid

train_set = data[:num_train]
y_train = labels[:num_train]
valid_set = data[num_train:num_train+num_valid]
y_valid = labels[num_train:num_train+num_valid]
test_set = data[-num_test:]
y_test = labels[-num_test:]

batch_size = 128
step_valid = 10
num_epoches = 101

model  = Pred(100,100,1,82, num_layer)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss = nn.MSELoss()



def train_iteration(batch_size,step,model, optimizer,loss,train_set,train_label):
  batch = train_set[step*batch_size:(step+1)*batch_size]
  groundtruth = train_label[step*batch_size:(step+1)*batch_size]
  optimizer.zero_grad()
  pred = model(batch).to(device)
  loss = loss(pred,numpy_to_tvar(groundtruth)).to(device)
  loss.backward()
  optimizer.step()
  
  return loss

def test_step(model, data, label_test, loss):
  pred_test = model(data).to(device)
  label_test = label_test
  error = loss(pred_test,numpy_to_tvar(label_test)).to(device)
  return error

def train(batch_size, model,optimizer, loss,train_set,
          y_train, valid_set, y_valid, step_valid,num_epoches):
  
  valid_results = []
  loss_batch = []
  min_error = 9999999999.0
  best_model = copy.deepcopy(model)
  num_steps = int(len(train_set)/batch_size)
  loss_epoches = []
  for e_i in range(num_epoches):
    loss_ei = 0.0
    for i in range(num_steps):
      step = i
      cost = train_iteration(batch_size,step,model,optimizer,loss,train_set,y_train)
      # train_results.append(cost)
      loss_ei += cost
      #print('step: ', iter_index, 'loss_per_batch: ', cost)
      
    if((e_i)%step_valid==0):
      error = test_step(model,valid_set,y_valid, loss)
      print('epoch: ', e_i,  'loss_valid: ', error)
      valid_results.append(error)
      if(min_error > error):
        best_model = copy.deepcopy(model)
        min_error = error
    loss_epoches.append(loss_ei/num_steps)
      
    print('------------------epoch: ', e_i, 'loss per epoch: ', loss_epoches[-1],' ------------------')
  return best_model, loss_epoches, valid_results

best_model, loss_batch, valid_results = train(batch_size, model, optimizer, loss,train_set, y_train,
                                              test_set, y_test, step_valid, num_epoches)

# Save model

torch.save(best_model.state_dict(),'newmodelRLSTM.h5')

# Plot loss function and predict values

plt.figure()
plt.plot(loss_batch, '-')
plt.legend('loss values')
plt.title('loss of training')
plt.savefig('loss_function_rlstm.png')
plt.close()

plt.figure()
plt.plot(valid_results, '-')
plt.legend('valid values')
plt.title('validation values')
plt.savefig('valid_rlstm.png')
plt.close()

gt = y_test
prediction = best_model(test_set)
test_acc = nn.L1Loss()(prediction, torch.FloatTensor(gt))
print('test accuracy: ', test_acc)
np.save('test_acc_rlstm.npy', test_acc)
plt.figure()
plt.plot(gt, 'r--')
plt.plot(prediction.detach().numpy(), 'b--')
plt.legend(('groundtruth', 'predict'))
plt.title('prediction test set')
plt.savefig('test_truth_rlstm.png')
plt.close()

np.save('loss_epochs_rlstm.npy',np.array(loss_batch))
np.save('valid_results_rlstm.npy',np.array(valid_results))
np.save('time_complexity_rlstm.npy',end-start)

# LOAD MODEL

new_model = Pred(100,100,1,82,num_layer) # chinh lai cac arguments
new_model.load_state_dict(torch.load('newmodelRLSTM.h5'))

# load params
for param_tensor in new_model.state_dict():
    print(param_tensor, "\t", new_model.state_dict()[param_tensor].size())
