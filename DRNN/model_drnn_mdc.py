import numpy as np
import tensorflow as tf
from DRNN_PYTORCH.drnnPytorch import multi_dRNN_with_dilations
from DRNN_PYTORCH.classification_modelsPytorch import _contruct_cells, _rnn_reformat
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
from torch.optim import optimizer
import copy










class DilatedRNN(nn.Module):
   def __init__(self, input_dims, hidden_structs, dilations, cells,num_layer):
        super(DilatedRNN,self).__init__()
        self.input_dims = input_dims
        self.hidden_structs = hidden_structs
        self.dilations = dilations
        self.cells = cells
        self.multi_dRNN_with_dilations = multi_dRNN_with_dilations(cells, dilations, num_layer) # num_layer dua vao de biet dang su
                                                                                                              # dung bao nhieu lop chu khong co 
                                                                                                              # dung trong ham
   def forward(self,x):
        """
            This function construct a multilayer dilated RNN for classifiction.
            Inputs:
                x -- a tensor of shape (batch_size, n_steps, input_dims).
                hidden_structs -- a list, each element indicates the hidden node dimension of each layer.
                dilations -- a list, each element indicates the dilation of each layer.
                n_steps -- the length of the sequence.
                n_classes -- the number of classes for the classification.
                input_dims -- the input dimension.
                cell_type -- the type of the RNN cell, should be in ["RNN", "LSTM", "GRU"].

            Outputs:
                pred -- the prediction logits at the last timestamp and the last layer of the RNN.
                        'pred' does not pass any output activation functions.
            """
        # error checking
        assert (len(self.hidden_structs) == len(self.dilations))
        # reshape inputs
        x_reformat = _rnn_reformat(x,self.input_dims)

        # define dRNN structures
        layer_outputs = self.multi_dRNN_with_dilations(x_reformat)
        
        return layer_outputs





