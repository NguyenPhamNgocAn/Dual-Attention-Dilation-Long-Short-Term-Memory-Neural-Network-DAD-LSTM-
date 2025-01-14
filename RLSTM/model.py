import numpy as np
import torch
import torch.nn as nn
from module_rlstm import RLSTM
from torch.autograd import Variable





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
    self.bias = Variable(
          torch.empty((hidden_size_2,)).normal_(mean=0,std=1),
                       requires_grad=True)
  def forward(self,inputs):
    x = torch.FloatTensor(inputs)
    lstm = self.lstm(x).view(-1,self.hidden_LSTM)# chi lay cell cuoi cung de predict
    dense_1 = self.dense_1(lstm)
    output = self.dense_2(dense_1) + self.bias
    return output
