import torch
import torch.nn as nn
import numpy as np
import copy
from torch.autograd import Variable
# VERSION : F(x) + X , residual block from bottom to top

class ResiLSTM(nn.Module):
  def __init__(self,seq, num_layer, input_size = 100 , hidden_size = 100):
    '''
        seq: number of step
        input_size: dim of input
        hidden_size: dim of hidden state
        num_layer: number of LSTM layer
    '''
    super(ResiLSTM,self).__init__()
    learn_models = []
    learn_models.append(nn.LSTM(input_size,hidden_size))
    self.hidden_size = hidden_size
    self.input_size = input_size
    self.seq = seq
    self.num_layer = num_layer
    # build iteration for following
    for i in range(num_layer - 1):
      learn_models.append(nn.LSTM(hidden_size, hidden_size))
    self.lstm = nn.ModuleList(learn_models)
    self.weight = nn.Parameter(
          torch.empty((seq,
                       hidden_size,
                       input_size)).normal_(mean=0,std=1),
                       requires_grad=True)
  def forward(self, inputs):
    # inputs : (batch_size, seq_len, num_feature)i

    x = inputs.permute(1,0,2)  # (seq_len, batch_size, num_feature)
    x= x.cuda(torch.device('cuda'))
    lstm,_ = self.lstm[0](x)
    for i in range(self.num_layer -1):
      lstm_backup = copy.copy(lstm)
      lstm,(h,c) = self.lstm[i+1](lstm) #( seq len, batch size, hidden size)

    Wio = self.lstm[-1].weight_ih_l0[3*self.hidden_size:].repeat(x.shape[0]*x.shape[1],1,1) # self.lstm2
    Wih = self.lstm[-1].weight_hh_l0[3*self.hidden_size:].repeat(x.shape[0]*x.shape[1],1,1)
    bio = self.lstm[-1].bias_ih_l0[3*self.hidden_size:].repeat(x.shape[0]*x.shape[1],1,1).view(-1,self.hidden_size,1)
    bih = self.lstm[-1].bias_hh_l0[3*self.hidden_size:].repeat(x.shape[0]*x.shape[1],1,1).view(-1, self.hidden_size,1)
    hh = torch.zeros_like(lstm)
    hh[1:] = lstm[:-1]
    hh = hh.view(-1,self.hidden_size,1)
    o_1 = torch.bmm(Wih,hh) +  torch.bmm(Wio,lstm_backup.view(-1,self.hidden_size,1)) + bio + bih
    o_1 = o_1.view(inputs.shape[0]*inputs.shape[1],self.hidden_size,1) ########### !!!!!!!!!!!!!
    lstm = lstm.view(inputs.shape[0]*inputs.shape[1],self.hidden_size,1)
    if ( self.input_size != self.hidden_size):
      weight = self.weight.repeat(inputs.shape[0],1,1,1).permute(1,0,2,3).contiguous().view(-1,
                                                              self.hidden_size,
                                                              self.input_size)
      lstm = lstm + o_1*(torch.bmm(weight,x.reshape(-1,x.shape[2],1)))
      lstm = lstm.view(inputs.shape[1], inputs.shape[0],-1)
    else:
      lstm = lstm + o_1*(x.reshape(-1,x.shape[2],1))
      lstm = lstm.view(inputs.shape[1], inputs.shape[0],-1) #(seq len, batch size, hidden size)
    return lstm

