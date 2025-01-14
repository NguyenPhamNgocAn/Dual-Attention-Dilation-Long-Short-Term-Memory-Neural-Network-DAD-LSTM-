import tensorflow as tf
from DRNN_PYTORCH.drnnPytorch import multi_dRNN_with_dilations
from torch.autograd import Variable
import torch
import numpy as np
import torch.nn as nn
def _contruct_cells(input_dims,hidden_structs, cell_type):
    """
    This function contructs a list of cells.
    """
    # error checking
    if cell_type not in ["RNN", "LSTM", "GRU"]:
        raise ValueError("The cell type is not currently supported.")

    # define cells
    cells = []
    # with tf.device('/gpu:0'):#####################################################################################
    input_dim = []
    input_dim.append(input_dims)
    for i in range(len(hidden_structs)):
        input_dim.append(hidden_structs[i])
    
    count  =  0
    for hidden_dims in hidden_structs:
        if cell_type == "RNN":
            cell = torch.nn.RNN(input_dim[count],hidden_dims)
        elif cell_type == "LSTM":
            cell =torch.nn.LSTM(input_dim[count],hidden_dims)
        elif cell_type == "GRU":
            cell = torch.nn.GRU(input_dim[count],hidden_dims)
        count =  count + 1
        cells.append(cell)
    return cells
# Khong can chuyen def _contruct_cells -> class _contruct_cells



def _rnn_reformat(x, input_dims):
    """
    This function reformat input to the shape that standard RNN can take.

    Inputs:
        x -- a tensor of shape (batch_size, n_steps, input_dims).
    Outputs:
        x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
    """
    x_ = torch.FloatTensor(x)
    # permute batch_size and n_steps
    x_ = x_.permute(1,0,2) #  x.permute(2, 0, 1) , x phai  = torch.tensor([])
    # reshape to (n_steps*batch_size, input_dims)
    x_ = torch.reshape(x_, [-1, input_dims]) # torch.reshape(a, (2, 2))
    # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
    x_reformat = torch.split(x_, np.shape(x)[0], 0) # torch.split(x_, batch_size, 0)

    return x_reformat

'''# chuyen def _rnn_reformat -> class _rnn_reformat
class _rnn_reformat():
    def __init__(self,input_dims):
        self.input_dims = input_dims

    def forward(self,x):
        """
        This function reformat input to the shape that standard RNN can take.

        Inputs:
            x -- a tensor of shape (batch_size, n_steps, input_dims).
        Outputs:
            x_reformat -- a list of 'n_steps' tenosrs, each has shape (batch_size, input_dims).
        """
        x_ = torch.tensor(x)
        # permute batch_size and n_steps
        x_ = x_.permute(1, 0, 2)  # x.permute(2, 0, 1) , x phai  = torch.tensor([])
        # reshape to (n_steps*batch_size, input_dims)
        x_ = torch.reshape(x_, [-1, self.input_dims])  # torch.reshape(a, (2, 2))
        # split to get a list of 'n_steps' tensors of shape (batch_size, input_dims)
        x_reformat = torch.split(x_, np.shape(x)[0], 0)  # torch.split(x_, batch_size, 0)

        return x_reformat


'''


def drnn_classification(x,
                        hidden_structs,
                        dilations,
                        n_steps,
                        n_classes,
                        input_dims=1, # input_dims=1 -> 6
                        cell_type="RNN"):
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
    assert (len(hidden_structs) == len(dilations))

    # reshape inputs
    x_reformat = _rnn_reformat(x, input_dims, n_steps)

    # construct a list of cells
    cells = _contruct_cells(input_dims,hidden_structs, cell_type)

    # define dRNN structures
    layer_outputs = multi_dRNN_with_dilations(cells, x_reformat, dilations)

    if dilations[0] == 1:
        # dilation starts at 1, no data dependency lost
        # define the output layer
        #with tf.device('/gpu:0'): ###########################################################################
        weights = Variable(torch.empty((hidden_structs[-1],n_classes)).normal_(mean=0,std=1), requires_grad=True)
        # torch.empty((hidden_structs[-1],n_classes)).normal_(mean=0,std=1)
        bias = Variable(torch.empty((n_classes,)).normal_(mean=0,std=1), requires_grad=True)
        # define prediction
        pred = torch.add(torch.matmul(layer_outputs[-1], weights),1, bias)
        # matmul is more general as depending on the inputs, it can correspond to dot, mm or bmm.
        # mm : matrix multiplication
        # torch.matmul(a,b)
        # torch.add(a,1,b)
    else:
        # dilation starts not at 1, needs to fuse the output

        # define output layer
        #with tf.device('/gpu:0'): ###############################################################################
        weights = Variable(torch.empty((hidden_structs[-1] * dilations[0], n_classes)).normal_(mean=0,std=1), requires_grad=True)
        bias = Variable(torch.empty((n_classes,)).normal_(mean=0,std=1),requires_grad=True)

        # concat hidden_outputs
        for idx, i in enumerate(range(-dilations[0], 0, 1)):
            if idx == 0:
                hidden_outputs_ = layer_outputs[i]
            else:
                hidden_outputs_ = torch.cat(
                    [hidden_outputs_, layer_outputs[i]],
                    axis=1)
                # torch.cat([hidden_outputs_, layer_outputs[i]],axis=1)

        pred = torch.add(torch.matmul(hidden_outputs_, weights),1, bias)

    return pred
