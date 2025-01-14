import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
from constants import device

from RLSTM.rlstm_predict import Pred

# from Residual_LSTM_best_regression import Pred 
def init_hidden(x, hidden_size: int):
    # x : tensor
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    # Variable has a Tensor type argument , it also used to auto back propagation since it has argument : require_grad
    # shape : (1,x.shape(0) <- batch size, hidden_state)
    return Variable(torch.zeros(1, x.size(0), hidden_size)).to(device)

import numpy as np
np.random.seed(1)

# Model
# inherited from Module class
class Encoder(nn.Module):
    # " : " symbol just make people know data type of arguments
    def __init__(self, input_size: int, hidden_size: int, T: int):

        """
        input size: number of underlying ( basis) factors (81)
        T: number of time steps (10)
        hidden_size: dimension of the hidden state
        """
        #  turn all of elements of Module class's init into this class's init
        # when calling father's functions , this merely use its graph but still use values of this current class
        super(Encoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1).to(device)
        # ////////////////////////////////////////////////////////////////////////////////////
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T - 1, out_features=1).to(device)
        # ////////////////////////////////////////////////////////////////////////////////////
    # in this code , input_data has numpy type , so at first we need to transfer into torch
    def forward(self, input_data):
        # input_data : input_size is the number of driving series , while each driving series contain 10 elements ,
        # the last one is y_label , so we use T-1 elements to guess the last one
        # at each time step , we use input_size elements  at correspond time step
        # input_data: (batch_size, T - 1, input_size)
        input_weighted = Variable(torch.zeros(input_data.size(0), self.T - 1, self.input_size)).to(device)
        input_encoded = Variable(torch.zeros(input_data.size(0), self.T - 1, self.hidden_size)).to(device)
        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(input_data, self.hidden_size)  # (1 ,batch_size, hidden_size)
        cell = init_hidden(input_data, self.hidden_size) # //

        for t in range(self.T - 1):
            # Eqn. 8: concatenate the hidden states with each predictor
            # at each time step , each driving series will be computed for each attention weight at each crorespond time step
            # so at each time step , we have input_size attention weights
            # hidden new  after permute : ( batch size * 81 * hidden size )
            # cell new after permute : //
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2) ,
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)), dim=2)  # (batch_size,input_size, 2*hidden_size + T - 1)
            # Eqn. 8: Get attention weights
            # applying formula (8) in paper , we multiply x for v.T ( a vector with shape ( (2*hidden_size + T - 1) * 1 )
            # infact it doesn't similar with Eqn 8 since this just concanate vectors and pass to a fully connected dim = 1
            # ( Eqn 8 in paper computed quite complexity )
            x = self.attn_linear(x.view(-1, self.hidden_size * 2 + self.T - 1))  # (batch_size * input_size) * 1
            # Eqn. 9: Softmax the attention weights
            # getting attention weights for elements of driving series (all) at correspond time step
            attn_weights = tf.softmax(x.view(-1, self.input_size), dim=1)  # (batch_size, input_size)
            # Eqn. 10: LSTM
            # multiplying attention weights for correspond elements of driving s
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)

            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            # this below code line just to fix warning when using data parallel for multi-gpus
            self.lstm_layer.flatten_parameters()
            # weighted_input.unsqueeze(0) : insert into weighted_input one dimensional ( first position )
            # _ : output of something with shape :( 1 , 128, 64 )
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden
            #print("hidden shape :" , hidden.shape)
            #print("cell shape : " , cell.shape)

        return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, hidden_LSTM, hidden_dense_1, hidden_dense_2,input_dim_rlstm,encoder_hidden_size: int, decoder_hidden_size: int, T: int,out_feats=1):# them tham so : hidden_LSTM,hidden_size_1, hidden_size 2, input_dim
        super(Decoder, self).__init__()
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size

        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1)).to(device)
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size).to(device)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats).to(device)
        self.fc.weight.data.normal_() 
        self.rlstm = Pred(hidden_LSTM= hidden_LSTM,hidden_size_1= hidden_dense_1,hidden_size_2= hidden_dense_2,input_dim= input_dim_rlstm,num_layer=2).to(torch.device('cpu'))
    def forward(self, input_encoded, y_history):
        #input_encoded: (batch_size, T - 1, encoder_hidden_size)
        #y_history: (batch_size, (T-1))
        #Initialize hidden and cell, (1, batch_size, decoder_hidden_size)
        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        context = Variable(torch.zeros(input_encoded.size(0), self.encoder_hidden_size)).to(device)

        for t in range(self.T - 1):
            # (batch_size, T-1, (2 * decoder_hidden_size + encoder_hidden_size))
            # this step is different from encoder since we were not compute attention weights for each elements in a step time
            # we computed attn weights for each h_t and then sum all to get context vecctor
            x = torch.cat((hidden.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T - 1, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2)
            # Eqn. 12 & 13: softmax on the computed attention weights
            x = tf.softmax(
                    self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T - 1),
                    dim=1)  # (batch_size, T - 1)

            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)

            # Eqn. 15
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1))  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters() #  remove warning
            # since hidden and cell are 3D , y_tile must be 3D
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # 1 * batch_size * decoder_hidden_size , type: torch.Tensor 
            cell = lstm_output[1]  # 1 * batch_size * decoder_hidden_size
	    # # Modify ####################################################################################################################### 
	    # # Method 1
	    # # reshape data
        hidden_reshape = hidden[0].unsqueeze(-1) # (batch_size, decoder_hidden_size,1)
        drnn_output = self.rlstm(hidden_reshape)
	    # # Method 2
	    # hidden_context = torch.cat((hidden[0],context), dim = 1 )
	    # drnn_output = self.drnn(hidden_context) # check shape (batch_size, 1)
	    # drnn_output = self.rlstm(hidden_context) 
	    #
	    #
        return drnn_output 



        


