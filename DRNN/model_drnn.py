import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data  # van chay duoc
from DRNN_PYTORCH.classification_modelsPytorch import _contruct_cells, _rnn_reformat
from DRNN_PYTORCH.drnnPytorch import multi_dRNN_with_dilations
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
from torch.optim import optimizer
import copy










class DilatedRNN(nn.Module):
   def __init__(self, n_steps, input_dims, n_classes, hidden_structs, dilations, cells, num_layer):
        super(DilatedRNN,self).__init__()
        self.n_steps = n_steps
        self.input_dims = input_dims
        self.n_classes = n_classes
        self.hidden_structs = hidden_structs
        self.dilations = dilations
        self.cells = cells
        self.multi_dRNN_with_dilations = multi_dRNN_with_dilations(self.cells,self.dilations, num_layer) # num_layer dua vao de biet dang su
                                                                                                              # dung bao nhieu lop chu khong co 
                                                                                                              # dung trong ham

        if dilations[0] == 1:
            # dilation starts at 1, no data dependency lost
            # define the output layer
            self.weights = nn.Parameter(Variable(torch.empty((hidden_structs[-1], n_classes)).normal_(mean=0, std=1), requires_grad=True))
            self.bias = nn.Parameter(Variable(torch.empty((n_classes,)).normal_(mean=0, std=1), requires_grad=True))
        else:
            # dilation starts not at 1, needs to fuse the output
            # define output layer
            self.weights = nn.Parameter(torch.tensor(np.random.normal(0,1,(hidden_structs[-1]*dilations[0], n_classes)),requires_grad=True))
            self.bias = nn.Parameter(torch.tensor(np.random.normal(0, 1, ( n_classes,)), requires_grad=True))

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
        if self.dilations[0] == 1:
            pred = torch.add(torch.matmul(layer_outputs[-1], self.weights), 1, self.bias)
            # matmul is more general as depending on the inputs, it can correspond to dot, mm or bmm.
            # mm : matrix multiplication
            # torch.matmul(a,b)
            # torch.add(a,1,b)
        else:
            # concat hidden_outputs
            for idx, i in enumerate(range(-self.dilations[0], 0, 1)):
                if idx == 0:
                    hidden_outputs_ = layer_outputs[i]
                else:
                    hidden_outputs_ = torch.cat(
                        [hidden_outputs_, layer_outputs[i]],
                        axis=1)
                    # torch.cat([hidden_outputs_, layer_outputs[i]],axis=1)

            pred = torch.add(torch.matmul(hidden_outputs_, self.weights), 1, self.bias)

        return pred





