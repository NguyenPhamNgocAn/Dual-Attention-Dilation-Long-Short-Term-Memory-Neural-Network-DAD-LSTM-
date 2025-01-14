import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as tf
import torch.nn as nn
from constants_1 import device
from DRNN_PYTORCH.model_drnn_mdc import DilatedRNN
from RLSTM.rlstm import ResiLSTM
def init_hidden(x, hidden_size: int):
    # x : tensor
    """
    Train the initial value of the hidden state:
    https://r2rt.com/non-zero-initial-states-for-recurrent-neural-networks.html
    """
    # Variable has a Tensor type argument , it also used to auto back propagation since it has argument : require_grad
    # shape : (1,x.shape(0) <- batch size, hidden_state)
    return Variable(torch.zeros(1, x.size(0), hidden_size)).to(device)


# Model
# inherited from Module class
class Encoder(nn.Module):
    # " : " symbol just make people know data type of arguments
    # "underscore"

    def __init__(self, input_size: int, hidden_size: int, T: int, hidden_structs, dilations, cells,num_layer, mode,num_layer_rlstm): # mode = [LSTM,RNN,NONE]

        """
        input size: number of underlying ( basis) factors (81)
        """
        # initialize all of elements of Module class's init into this class's init
        # when calling father's functions , this merely use its graph but still use values of this current class
        super(Encoder, self).__init__()


        self.input_size = input_size
        self.hidden_size = hidden_size
        self.T = T
        self.tanh = nn.Tanh()
        self.sm = nn.Softmax(dim = 1)
        self.hidden_structs = hidden_structs
        self.dilations = dilations
        self.cells = cells
        self.num_layer = num_layer
        self.mode = mode

        self.lstm_layer = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1).to(device)
        
        self.attn_linear = nn.Linear(in_features=2 * hidden_size + T, out_features=1).to(device)
        
        self.attn_linear2 = nn.Linear(in_features= T, out_features=1).to(device)
        self.linear = nn.Linear(in_features=2 * hidden_size + T, out_features= T).to(device)
        self.drnn = DilatedRNN(hidden_size, hidden_structs, dilations, cells,num_layer).to(torch.device('cpu'))
        self.rlstm = ResiLSTM(T, num_layer_rlstm,hidden_size, hidden_size)
    # in this code , input_data has numpy type , so at first we need to transfer into torch
    def forward(self, input_data):
        # input_data : input_size is the number of driving series , while each driving series contain 10 elements ,
        # the last one is y_label , so we use T-1 elements to guess the last one
        # at each time step , we use input_size elements  at correspond time step
        # input_data: (batch_size, T, input_size)
        if(self.mode.lower() not in ["drnn","rlstm","none"]):
            print("mode doesnt exist")
            exit()

        # last layer of hidden_structs  must equal to hidden_size
        if (self.hidden_structs[-1] != self.hidden_size):
            print('dimensional doesnt match between hidden_structs, encoder_hidden_size')
            exit()
        output_encoded = torch.zeros((self.T, input_data.shape[0], self.hidden_size))
        # bo variable 2 cai duoi
        input_weighted = torch.zeros((input_data.size(0), self.T, self.input_size))
        input_encoded = torch.zeros((input_data.size(0), self.T, self.hidden_size))
        # hidden, cell: initial states with dimension hidden_size
        hidden = init_hidden(input_data, self.hidden_size)# (1, batch_size, hidden_size)
        cell = init_hidden(input_data, self.hidden_size) #
        # create bias
	# Sua lai T-1 -> T
	# dua LSTM xuong cuoi
	# dong tanh(x), dem len phan khai bao
	# 
        # bias2 = Variable(torch.zeros(input_data.size(0), self.T - 1, 1))  # (batch size, T-1,1)
        for t in range(self.T):
            # Eqn. 8: concatenate the hidden states with each predictor
            # at each time step , each driving series will be computed for each attention weight at each crorespond time step
            # so at each time step , we have input_size attention weights
            # hidden new  after permute : ( batch size * 81 * hidden size )
            # cell after permute : //
            
            x = torch.cat((hidden.repeat(self.input_size, 1, 1).permute(1, 0, 2) ,
                           cell.repeat(self.input_size, 1, 1).permute(1, 0, 2),
                           input_data.permute(0, 2, 1)),dim=2)  # batch_size , input_size ,2*hidden_size + T
            

	    # Eqn. 8: Get attention weights
            # applying formula (8) in paper , we multiply x for v.T ( a vector with shape ( (2*hidden_size + T - 1) * 1 )
            # infact it doesn't similar with Eqn 8 since this just concanate vectors and pass to a fully connected dim = 1
            # ( Eqn 8 in paper computed quite complexity )
            x = self.linear(x.view(-1, self.hidden_size * 2 + self.T))  # (batch_size x input_size , T)
            x = self.tanh(x)  
            x = self.attn_linear2(x)
	    # Eqn. 9: Softmax the attention weights
            # getting attention weights for elements of driving series (all) at correspond time step
            attn_weights = self.sm(x.view(-1, self.input_size))  # (batch_size, input_size)
            

	    # Eqn. 10: multiply attention weight	
            # multiplying attention weights for correspond elements of driving s
            weighted_input = torch.mul(attn_weights, input_data[:, t, :])  # (batch_size, input_size)

            # Fix the warning about non-contiguous memory
            # see https://discuss.pytorch.org/t/dataparallel-issue-with-flatten-parameter/8282
            # this below code line just to fix warning when using data parallel for multi-gpus
	    
	    
            # weighted_input.unsqueeze(0) : insert into weighted_input one dimensional ( first position )
            # _ : output of something with shape :( 1 , 128, 64 )
            self.lstm_layer.flatten_parameters()
            _, lstm_states = self.lstm_layer(weighted_input.unsqueeze(0), (hidden, cell))
            hidden = lstm_states[0]
            cell = lstm_states[1]
            # Save output
            input_weighted[:, t, :] = weighted_input
            input_encoded[:, t, :] = hidden[0]
        
        if(self.mode == 'drnn'):
            # input_encoded (batch_size, T, hidden_output)
            output =self.drnn(input_encoded)
            for step in range(len(output)):
                output_encoded[step,:,:] = output[step]

            return input_weighted, output_encoded.permute(1,0,2)
        elif(self.mode == 'rlstm'):
            output = self.rlstm(input_encoded)
            return input_weighted, output.permute(1,0,2)
        else:
            return input_weighted, input_encoded


class Decoder(nn.Module):

    def __init__(self, encoder_hidden_size: int, decoder_hidden_size: int, T: int, hidden_structs, dilations, cells,num_layer, mode,
            num_layer_rlstm,enc_size=81, out_feats=1):
        super(Decoder, self).__init__()
        # enc_size: number of stock
        self.T = T
        self.encoder_hidden_size = encoder_hidden_size
        self.decoder_hidden_size = decoder_hidden_size
        self.sm = nn.Softmax(dim = 1)
        self.attn_layer = nn.Sequential(nn.Linear(2 * decoder_hidden_size + encoder_hidden_size,
                                                  encoder_hidden_size),
                                        nn.Tanh(),
                                        nn.Linear(encoder_hidden_size, 1)).to(device)
        self.lstm_layer = nn.LSTM(input_size=out_feats, hidden_size=decoder_hidden_size).to(device)
        self.lstm_layer2 = nn.LSTM(input_size=enc_size, hidden_size=encoder_hidden_size).to(device)
        self.fc = nn.Linear(encoder_hidden_size + out_feats, out_feats).to(device)
        self.fc_final = nn.Linear(decoder_hidden_size,out_feats).to(device)
        self.bias = nn.Parameter(torch.empty((out_feats,)).normal_(mean=0,std=1))
        self.bias2 = nn.Parameter(torch.empty((out_feats,)).normal_(mean=0,std=1))
        self.bias3 = nn.Parameter(torch.empty((decoder_hidden_size,)).normal_(mean=0,std=1))
        self.fc.weight.data.normal_()
        self.linear = nn.Linear(2*decoder_hidden_size + encoder_hidden_size, decoder_hidden_size).to(device)
        self.linear2 = nn.Linear(decoder_hidden_size, out_feats).to(device)
        self.linear3 = nn.Linear(decoder_hidden_size+ encoder_hidden_size, decoder_hidden_size).to(device)
        self.drnn2 = DilatedRNN(decoder_hidden_size, hidden_structs, dilations, cells,num_layer).to(torch.device('cpu'))
        self.rlstm2 = ResiLSTM(T, num_layer_rlstm, decoder_hidden_size, decoder_hidden_size)
        self.hidden_structs = hidden_structs
        self.dilations = dilations
        self.cells = cells
        self.num_layer = num_layer
        self.num_layer_rlstm  = num_layer_rlstm
        self.mode = mode
    def forward(self,input_encoded, y_history):#//////////////////////
        # input_encoded: (batch_size, T, encoder_hidden_size)
        # y_history: (batch_size,T-1)
        # Initialize hidden and cell, (1, batch_size, decoder_hidden_size)

	# Bo bias3
	# chuyen bias len khai bao
	# decoder giu nguyen T-1
        input_encoded = input_encoded.cuda(torch.device('cuda'))
        if(self.mode.lower() not in ["drnn","rlstm","none"]):
            print("mode doesnt exist")
            exit()
        hidden_states = torch.zeros((self.T-1, input_encoded.shape[0], self.decoder_hidden_size))

        if (self.hidden_structs[-1] != self.decoder_hidden_size):
            print('dimensional doesnt match between hidden_structs, decoder_hidden_size')
            exit()

        hidden = init_hidden(input_encoded, self.decoder_hidden_size)
        cell = init_hidden(input_encoded, self.decoder_hidden_size)
        # bo Variable
        context = torch.zeros((input_encoded.size(0), self.encoder_hidden_size))
        #hiddenenc = Variable(torch.zeros(input_encoded.size(0), self.T - 1, self.encoder_hidden_size)) # initialize tensor to store hidden state after extract feature
        # from weighted data
        # from weighted output of encoder, compute hidden state ( h1, h2,..., hT)
        # hiddenenc : ( batch size,T-1,encoder_hidden_size)
        #hiddendec = init_hidden(input_encoded, self.decoder_hidden_size)
        #celldec = init_hidden(input_encoded, self.decoder_hidden_size)
        for t in range(self.T - 1):
            # (batch_size, T, (2 * decoder_hidden_size + encoder_hidden_size))
            # this step is different from encoder since we were not compute attention weights for each elements in a step time
            # we computed attn weights for each h_t and then sum all to get context vecctor
	    # Eq 12
            x = torch.cat((hidden.repeat(self.T, 1, 1).permute(1, 0, 2),
                           cell.repeat(self.T, 1, 1).permute(1, 0, 2),
                           input_encoded), dim=2) 
            x = self.sm(
                    self.attn_layer(
                        x.view(-1, 2 * self.decoder_hidden_size + self.encoder_hidden_size)
                    ).view(-1, self.T)) # (batch size, T)
            # Eqn. 14: compute context vector
            context = torch.bmm(x.unsqueeze(1), input_encoded)[:, 0, :]  # (batch_size, encoder_hidden_size)
            # Eqn. 15
            # add bias extra
            y_tilde = self.fc(torch.cat((context, y_history[:, t]), dim=1)) + self.bias  # (batch_size, out_size)
            # Eqn. 16: LSTM
            self.lstm_layer.flatten_parameters() #  remove warning
            # since hidden and cell are 3D , y_tile must be 3D
            _, lstm_output = self.lstm_layer(y_tilde.unsqueeze(0), (hidden, cell))
            hidden = lstm_output[0]  # (1 , batch_size , decoder_hidden_size)
            cell = lstm_output[1]    # (1 , batch_size , decoder_hidden_size)
            hidden_states[t,:,:] = hidden[0]

        if(self.mode == 'drnn'):
            hidden_states = hidden_states.permute(1,0,2)
            output_drnn = self.drnn2(hidden_states) #(T-1, Batch_size, decoder_hidden_size)
            pre_out = output_drnn[-1] #(batch size, decoder_hidden_size)
        elif(self.mode =='rlstm'):
            hidden_states = hidden_states.permute(1,0,2)
            output_rlstm = self.rlstm2(hidden_states) #(T-1, Batch_size, decoder_hidden_size)
            pre_out = output_rlstm[-1]
        else:
            pre_out = self.linear3(torch.cat((hidden[0], context),dim = 1)) + self.bias3
        # Eqn. 22: final output
        return self.fc_final(pre_out) + self.bias2
        


