import sys

sys.path.append("./models")
import numpy as np
import tensorflow as tf
from classification_modelsPytorch_copy import _contruct_cells, _rnn_reformat
from drnnPytorch import multi_dRNN_with_dilations
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
from torch import nn
from torch.optim import optimizer
import copy
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
# gpus = tf.config.experimental.list_logical_devices('GPU')

print('check if GPU is available')
print(torch.cuda.is_available())


start = time.time()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def MAPELoss(output, target):
    return torch.mean(torch.abs((target-output)/target))

# configuration
n_steps = 10
input_dims = 82  # -> 6
n_classes = 1  # -> 1

# model config
cell_type = "LSTM"
assert (cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [20] * 4
dilations = [1, 2, 4, 8] # 4 layers 
assert (len(hidden_structs) == len(dilations))
num_layer = 4


# Build Model
cells = _contruct_cells(input_dims,hidden_structs,cell_type)
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

################################################################## CREATE PROGRAM FOR TRAING  ###################################################################
data = pd.read_csv('nasdaq100_padding.csv')
stocks_name = list(data.columns)
# Preprocessing
scale = StandardScaler().fit(data)
proc_dat = scale.transform(data)

print('data shape: ',proc_dat.shape)
steps = 10
name_stock = "NDX"
databatch = [] # luu cac batch , size = (num of batch,steps,6)
labels = []

index_stock = stocks_name.index(name_stock)
target = proc_dat[:,index_stock]
arr = np.array(proc_dat)


for i in range(len(arr)-steps):
    databatch.append(arr[i:i+steps])
    labels.append(target[i+steps])
# save file numpy .npy
np.save('nasdaq.npy',databatch)
np.save('lb_nasdaq.npy',labels)

# load file numpy .npy


data = np.load('nasdaq.npy')

labels = np.load('lb_nasdaq.npy')
print('data shape after processing: ',np.shape(data))
print('labels of data: ',np.shape(labels))


num_data = np.shape(data)[0]
num_train =  35090#int(num_data*0.65)
num_valid = 2730 #int(num_data*0.15)
num_test = 2730 #num_data - num_train - num_valid

train_set = data[:num_train]
y_train = labels[:num_train]
valid_set = data[num_train:num_train+num_valid]
y_valid = labels[num_train:num_train+num_valid]
test_set = data[-num_test:]
y_test = labels[-num_test:]

model = DilatedRNN(n_steps,input_dims, n_classes, hidden_structs, dilations, cells, num_layer = num_layer)
optimizer = torch.optim.RMSprop(model.parameters() , lr=0.001, alpha=0.90)
loss = nn.MSELoss()

batch_size = 128
step_valid = 3
num_epoches = 43

def train_iteration(batch_size,step,model, optimizer,loss,train_set,train_label):
  batch = train_set[step*batch_size:(step+1)*batch_size]
  groundtruth = torch.FloatTensor(train_label[step*batch_size:(step+1)*batch_size])
  optimizer.zero_grad()
  pred = model(batch)
  loss = loss(pred,groundtruth)
  loss.backward()
  optimizer.step()

  return loss


def test_step(model, data, label_test, loss):
  pred_test = model(data)
  label_test = torch.FloatTensor(label_test)
  error = loss(pred_test,label_test)
  return error


def train(batch_size, model,optimizer, loss,train_set,
          y_train, valid_set, y_valid, step_valid,num_epoches):

  valid_results = []
  loss_batch = []
  min_error = 9999999999.0
  best_model = copy.deepcopy(model)
  num_steps = int(len(train_set)/batch_size)
  loss_epoches = []
  for e_i in range(num_epoches):
    loss_ei = 0.0
    for i in range(num_steps):
      step = i
      cost = train_iteration(batch_size,step,model,optimizer,loss,train_set,y_train)
      # train_results.append(cost)
      loss_ei += cost
      #print('step: ', iter_index, 'loss_per_batch: ', cost)

    if((e_i)%step_valid==0):
      error = test_step(model,valid_set,y_valid, loss)
      #print('epoch: ', e_i,  'loss_valid: ', error)
      valid_results.append(error)
      if(min_error > error):
        best_model = copy.deepcopy(model)
        min_error = error
    loss_epoches.append(loss_ei/num_steps)
    '''if(min_error > loss_epoches[-1]):
        best_model = copy.deepcopy(model)
        min_error = loss_epoches[-1]
    '''
    print('------------------epoch: ', e_i, 'loss per epoch: ', loss_epoches[-1],' ------------------')
  return best_model, loss_epoches, valid_results


###################################################### TRAINING #################################################
best_model, loss_batch, valid_results = train(batch_size, model, optimizer, loss,train_set, y_train,
         test_set, y_test, step_valid, num_epoches)

torch.save(best_model.state_dict(),'modelRLSTM.h5')

end = time.time()
# Plot loss function and predict values

plt.figure()
plt.plot(loss_batch, '-')
plt.legend('loss values')
plt.title('loss of training DRNN')
plt.savefig('loss_training_drnn.png')
plt.close()

gt = y_test
prediction = best_model(test_set)
test_acc = nn.L1Loss()(prediction, torch.FloatTensor(gt))
print('test accuracy: ', test_acc)
np.save('acc_drnn.npy', test_acc.detach().numpy())
print('time: ', end-start)
np.save('time_complexity.npy', end-start)
plt.figure()
plt.plot(gt, 'r--')
plt.plot(prediction.detach().numpy(), 'b--')
plt.legend(('groundtruth', 'predict'))
plt.title('prediction test set')
plt.savefig('test_truth_drnn.png')
plt.close()


plt.figure()
plt.plot(valid_results, 'r-')
plt.legend('valid values')
plt.title('validation values')
plt.savefig('valid_drnn.png')
plt.close()


print('loss function: ', loss_batch)
print('valid values: ', valid_results)

