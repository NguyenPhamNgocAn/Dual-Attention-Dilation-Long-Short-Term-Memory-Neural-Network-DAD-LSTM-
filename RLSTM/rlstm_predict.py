import torch
import torch.nn as nn
import numpy as np
import copy
from constants import device





class RLSTM(nn.Module):
  def __init__(self,num_layer,input_size = 100 , hidden_size = 100):
    super(RLSTM,self).__init__()
    learn_models = []
    learn_models.append(nn.LSTM(input_size,hidden_size))
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.num_layer = num_layer
    # build iteration for following 
    for i in range(num_layer - 1):
      learn_models.append(nn.LSTM(hidden_size, hidden_size))
    self.rlstm = nn.ModuleList(learn_models)
  def forward(self, inputs):
    # inputs : (batch_size, seq_len, num_feature)
    print(inputs.shape)
    x = np.transpose(np.array(inputs.detach()),(1,0,2))  # (seq_len, batch_size, num_feature) 
    print('finish')
    print('type: ', type(x))
    x = torch.FloatTensor(x)
    lstm,_ = self.rlstm[0](x)
    for i in range(self.num_layer -1):
      lstm_backup = copy.copy(lstm) 
      lstm,(h,c) = self.rlstm[i+1](lstm)
      Wio = self.rlstm[i+1].weight_ih_l0[3*self.hidden_size:].repeat(x.shape[1],1,1) # self.lstm2
      Wih = self.rlstm[i+1].weight_hh_l0[3*self.hidden_size:].repeat(x.shape[1],1,1)
      bio = self.rlstm[i+1].bias_ih_l0[3*self.hidden_size:].repeat(x.shape[1],1,1).view(-1,self.hidden_size,1)
      bih = self.rlstm[i+1].bias_hh_l0[3*self.hidden_size:].repeat(x.shape[1],1,1).view(-1, self.hidden_size,1)
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
    self.num_layer = num_layer
    self.rlstm = RLSTM(num_layer,input_dim, hidden_LSTM).to(device)
    self.dense_1 = nn.Linear(hidden_LSTM,hidden_size_1).to(device)
    self.dense_2 = nn.Linear(hidden_size_1,hidden_size_2).to(device)
    

  def forward(self,inputs):
    inputs = inputs.cpu()
    x = torch.FloatTensor(inputs)
    rlstm = self.rlstm(x).view(-1,self.hidden_LSTM) # chi lay cell cuoi cung de predict

    dense_1 = self.dense_1(rlstm)
    output = self.dense_2(dense_1)
    return output


