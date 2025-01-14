import torch.nn as nn
import torch
import numpy as np
import copy 


class RLSTM(nn.Module):
  def __init__(self,input_size = 100 , hidden_size = 100, num_layer=2):
    super(RLSTM,self).__init__()
    learn_models = []
    learn_models.append(nn.LSTM(input_size,hidden_size))
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.num_layer = num_layer
    # build iteration for following 
    for i in range(num_layer - 1):
      learn_models.append(nn.LSTM(hidden_size, hidden_size))
    self.lstm = nn.ModuleList(learn_models)
  def forward(self, inputs):
    # inputs : (batch_size, seq_len, num_feature)
    x = np.transpose(inputs,(1,0,2))  # (seq_len, batch_size, num_feature)
    lstm,_ = self.lstm[0](x)
    for i in range(self.num_layer -1):
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
