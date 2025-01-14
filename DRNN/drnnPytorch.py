import copy
import itertools
import numpy as np
import torch
import tensorflow as tf
import torch.nn as nn
"""def dRNN(cell, inputs, rate, scope='default'):
    '''
    This function constructs a layer of dilated RNN.
    Inputs:
        cell -- the dilation operations is implemented independent of the RNN cell.
            In theory, any valid tensorflow rnn cell should work.
        inputs -- the input for the RNN. inputs should be in the form of
            a list of 'n_steps' tenosrs. Each has shape (batch_size, input_dims)
        rate -- the rate here refers to the 'dilations' in the orginal WaveNet paper.
        scope -- variable scope.
    Outputs:
        outputs -- the outputs from the RNN.
    '''
    n_steps = len(inputs)
    if rate < 0 or rate >= n_steps:
        raise ValueError('The \'rate\' variable needs to be adjusted.')
        print('rate: ', rate)
    print( "Building layer: %s, input length: %d, dilation rate: %d, input dim: %d." % (
        scope, n_steps, rate, inputs[0].shape[1]))

    # make the length of inputs divide 'rate', by using zero-padding
    EVEN = (n_steps % rate) == 0
    if not EVEN:
        # Create a tensor in shape (batch_size, input_dims), which all elements are zero.
        # This is used for zero padding
        zero_tensor = torch.zeros_like(inputs[0])
        dialated_n_steps = n_steps // rate + 1
        print( "=====> %d time points need to be padded. " % (
            dialated_n_steps * rate - n_steps))
        print( "=====> Input length for sub-RNN: %d" % (dialated_n_steps))
        for i_pad in range(dialated_n_steps * rate - n_steps):
            inputs.append(zero_tensor)
    else:
        dialated_n_steps = n_steps // rate
        print( "=====> Input length for sub-RNN: %d" % (dialated_n_steps))

    # now the length of 'inputs' divide rate
    # reshape it in the format of a list of tensors
    # the length of the list is 'dialated_n_steps'
    # the shape of each tensor is [batch_size * rate, input_dims]
    # by stacking tensors that "colored" the same

    # Example:
    # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
    # zero-padding --> [x1, x2, x3, x4, x5, 0]
    # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
    # which the length is the ceiling of n_steps/rate
    dilated_inputs = [torch.cat(inputs[i * rate:(i + 1) * rate],0) for i in range(dialated_n_steps)]
    dilated_outputs = []
    ####################################################
    _, dilated_output = cell(torch.unsqueeze(dilated_inputs[0],0))
    dilated_outputs.append(torch.squeeze(dilated_output,0))
    if( len(dilated_inputs) >1):
        for step in range(1,len(dilated_inputs)):
            _, dilated_output = cell(torch.unsqueeze(dilated_inputs[step],0),dilated_output)
            dilated_outputs.append(torch.squeeze(dilated_output,0))
            # building a dialated RNN with reformated (dilated) inputs

    ###################################################


    # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
    # split each element of the outputs from size [batch_size*rate, input_dims] to
    # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
    splitted_outputs = [torch.split(output, inputs[0].shape[0], 0)
                        for output in dilated_outputs]
    unrolled_outputs = [output
                        for sublist in splitted_outputs for output in sublist]
    # remove padded zeros
    outputs = unrolled_outputs[:n_steps]


    return outputs
"""



# Chuyen def dRNN thanh class

class dRNN(nn.Module):
    def __init__(self,cell,rate,scope='default'):
        super(dRNN,self).__init__()
        self.cell = cell
        self.rate = rate
        self.scope = scope
    def forward(self, inputs):
        # them vao doi kie du lieu
        inputs = list(inputs)
        n_steps = len(inputs)
        if self.rate < 0 or self.rate >= n_steps:
            print('rate: ', self.rate)
            print("length of inputs: ", n_steps)
            raise ValueError('The \'rate\' variable needs to be adjusted.')
           
      # print("Building layer: %s, input length: %d, dilation rate: %d, input dim: %d." % (
      #      self.scope, n_steps, self.rate, inputs[0].shape[1]))

        # make the length of inputs divide 'rate', by using zero-padding
        EVEN = (n_steps % self.rate) == 0
        if not EVEN:
            # Create a tensor in shape (batch_size, input_dims), which all elements are zero.
            # This is used for zero padding
            zero_tensor = torch.zeros_like(inputs[0])
            dialated_n_steps = n_steps // self.rate + 1
       #     print("=====> %d time points need to be padded. " % (
       #             dialated_n_steps * self.rate - n_steps))
       #     print("=====> Input length for sub-RNN: %d" % (dialated_n_steps))
            for i_pad in range(dialated_n_steps * self.rate - n_steps):
                inputs.append(zero_tensor)
        else:
            dialated_n_steps = n_steps // self.rate
        #  print("=====> Input length for sub-RNN: %d" % (dialated_n_steps))

        # now the length of 'inputs' divide rate
        # reshape it in the format of a list of tensors
        # the length of the list is 'dialated_n_steps'
        # the shape of each tensor is [batch_size * rate, input_dims]
        # by stacking tensors that "colored" the same

        # Example:
        # n_steps is 5, rate is 2, inputs = [x1, x2, x3, x4, x5]
        # zero-padding --> [x1, x2, x3, x4, x5, 0]
        # we want to have --> [[x1; x2], [x3; x4], [x_5; 0]]
        # which the length is the ceiling of n_steps/rate 
        dilated_inputs = torch.zeros((dialated_n_steps, inputs[0].shape[0]*self.rate, inputs[0][0].shape[0]))
        #dilated_inputs = [torch.cat(inputs[i * self.rate:(i + 1) * self.rate], 0) for i in range(dialated_n_steps)]
        for i in range(dialated_n_steps):
            dilated_inputs[i] = torch.cat(inputs[i * self.rate:(i + 1) * self.rate], 0)
        '''dilated_outputs = []
        ####################################################
        _, dilated_output = self.cell(torch.unsqueeze(dilated_inputs[0], 0))
        dilated_outputs.append(torch.squeeze(dilated_output, 0))
        if (len(dilated_inputs) > 1):
            for step in range(1, len(dilated_inputs)):
                _, dilated_output = self.cell(torch.unsqueeze(dilated_inputs[step], 0), dilated_output)
                dilated_outputs.append(torch.squeeze(dilated_output, 0))
                # building a dialated RNN with reformated (dilated) inputs
        '''
        ###################################################

        dilated_outputs,_ = self.cell(dilated_inputs.cuda(torch.device('cuda')))

        # reshape output back to the input format as a list of tensors with shape [batch_size, input_dims]
        # split each element of the outputs from size [batch_size*rate, input_dims] to
        # [[batch_size, input_dims], [batch_size, input_dims], ...] with length = rate
        splitted_outputs = [torch.split(output, inputs[0].shape[0], 0)
                            for output in dilated_outputs]
        unrolled_outputs = [output
                            for sublist in splitted_outputs for output in sublist]
        # remove padded zeros
        outputs = unrolled_outputs[:n_steps]

        return outputs



'''def multi_dRNN_with_dilations(cells, inputs, dilations):
    """
    This function constucts a multi-layer dilated RNN.
    Inputs:
        cells -- A list of RNN cells.
        inputs -- A list of 'n_steps' tensors, each has shape (batch_size, input_dims).
        dilations -- A list of integers with the same length of 'cells' indicates the dilations for each layer.
    Outputs:
        x -- A list of 'n_steps' tensors, as the outputs for the top layer of the multi-dRNN.
    """
    assert (len(cells) == len(dilations))
    x = copy.copy(inputs)
    for cell, dilation in zip(cells, dilations):
        scope_name = "multi_dRNN_dilation_%d" % dilation
        x = dRNN(cell, x, dilation, scope=scope_name)
    return x
'''
# Chuyen def multi_dRNN_with_dilations thanh class multi_dRNN_with_dilations


'''class multi_dRNN_with_dilations(nn.Module):
    def __init__(self,cells,dilations):
        super(multi_dRNN_with_dilations,self).__init__()
        self.cells = cells
        self.dilations = dilations
        self.store = []
        assert (len(self.cells) == len(self.dilations))
        for cell, dilation in zip(self.cells,self.dilations):
            scope_name = "multi_dRNN_dilation_%d" % dilation
            self.store.append(dRNN(cell,dilation,scope= scope_name))

    def forward(self, inputs):
        """
            This function constucts a multi-layer dilated RNN.
            Inputs:
                cells -- A list of RNN cells.
                inputs -- A list of 'n_steps' tensors, each has shape (batch_size, input_dims).
                dilations -- A list of integers with the same length of 'cells' indicates the dilations for each layer.
            Outputs:
                x -- A list of 'n_steps' tensors, as the outputs for the top layer of the multi-dRNN.
            """

        x = copy.copy(inputs)
        for cell in range(len(self.cells)):
            x = self.store[cell](x)
        return x
'''


class multi_dRNN_with_dilations(nn.Module):
    def __init__(self,cells,dilations,num_layer):
        super(multi_dRNN_with_dilations,self).__init__()
        self.cells = cells
        self.dilations = dilations
        self.num_layer = num_layer
        layers = [] 
        assert (len(self.cells) == len(self.dilations))
        assert (len(self.cells) == self.num_layer)
        for layer in range(num_layer):
            layers.append(dRNN(cells[layer],dilations[layer],scope= "multi_dRNN_dilation_" +str(layer)))
        self.layers = nn.ModuleList(layers)
    
    def forward(self, inputs):
        """
            This function constucts a multi-layer dilated RNN.
            Inputs:
                cells -- A list of RNN cells.
                inputs -- A list of 'n_steps' tensors, each has shape (batch_size, input_dims).
                dilations -- A list of integers with the same length of 'cells' indicates the dilations for each layer.
            Outputs:
                x -- A list of 'n_steps' tensors, as the outputs for the top layer of the multi-dRNN.
            """
            
        x = copy.copy(inputs)
        for layer in range(self.num_layer):
            x = self.layers[layer](x)
       
        
        return x


