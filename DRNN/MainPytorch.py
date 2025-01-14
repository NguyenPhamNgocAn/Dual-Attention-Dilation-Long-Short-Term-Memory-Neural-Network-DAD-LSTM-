import sys

sys.path.append("./models")
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
# gpus = tf.config.experimental.list_logical_devices('GPU')

print('check if GPU is available')
print(torch.cuda.is_available())

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# configurations
data_dir = "./MNIST_data"  # -> change path
n_steps = 28 * 28  # -> 20
input_dims = 1  # -> 6
n_classes = 10  # -> 1

# model config
cell_type = "RNN"
assert (cell_type in ["RNN", "LSTM", "GRU"])
hidden_structs = [20] * 8
dilations = [1, 2, 4, 8, 16, 32, 64, 128] # 9 layers 
assert (len(hidden_structs) == len(dilations))
num_layer = 8 
seed = 92916
mnist = input_data.read_data_sets(data_dir, one_hot=True)
# learning config
batch_size = 64  # nen doi lai nho hon vi data it
learning_rate = 1.0e-2
#training_iters = batch_size * 19  # ////////
#testing_step = 10
#display_step = 10
#num_elements = len(mnist.test.labels)
#num_iters = int(num_elements/batch_size)
num_iters = 3
# permutation seed


print(' In this code, we use testing set instead of training set because the size of training set too large to train model ')
print(" number of train image: ", len(mnist.train.images))
print(" number of test image: ", len(mnist.test.images))
print(" number of validation image: ", len(mnist.validation.images))
print(" number of iter: ", num_iters) 

'''if 'seed' in globals():
    rng_permute = np.random.RandomState(seed)
    idx_permute = rng_permute.permutation(n_steps)
else:
    idx_permute = np.random.permutation(n_steps)
   ''' 
# build computation graph
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



# **************************************** CREATE CROSS VALIDATION ********************************** #

# NOTE: using testing set to substitude for training set because of the excessively memory usage 

batch_list_images = np.array_split(mnist.test.images,num_iters)
batch_list_labels = np.array_split(mnist.test.labels,num_iters)

folds_images_train = []
folds_labels_train = []
folds_images_test  = []
folds_labels_test  = []


for i in range(num_iters):
  folds_labels_train.append(np.delete(batch_list_labels,i,0))
  folds_images_train.append(np.delete(batch_list_images,i,0))
  folds_labels_test.append(batch_list_labels[i])
  folds_images_test.append(batch_list_images[i])
  
  


#step = 0
#train_results = []
#validation_results = []
#test_results = []
'''batch_x, batch_y = mnist.train.next_batch(batch_size)
batch_x = batch_x[:, idx_permute]
batch_x = batch_x.reshape([batch_size, n_steps, input_dims])
batch_x = torch.tensor(batch_x)'''

DRNN = DilatedRNN(n_steps,input_dims, n_classes, hidden_structs, dilations, cells, num_layer = num_layer)
optimizer = torch.optim.RMSprop(DRNN.parameters() , lr=0.001, alpha=0.90)
loss = nn.CrossEntropyLoss()
# using validation set for testing set because testing set is used for training step
#batch_x_valid = mnist.validation.images[0:500]
#batch_y_valid = mnist.validation.labels[0:500]
#batch_y_valid = torch.tensor(batch_y_valid,device=device)
#_, batch_y_valid = torch.max(batch_y_valid, -1)



# training for each iter
def train_iteration(batch_train, batch_label,device,n_steps, input_dims, model,optimizer, loss):
  batch_label = torch.tensor(batch_label,device = device)
  _, batch_label = torch.max(batch_label,-1)
  #batch_train = batch_train[:, idx_permute]
  batch_train = batch_train.reshape([-1, n_steps, input_dims])
  batch_train = torch.tensor(batch_train)
  pred = model(batch_train).to(device)
 
  optimizer.zero_grad()
  cost_ = loss(pred,batch_label).to(device) # loss function
  
  cost_.backward()
  optimizer.step()
  
  _,label = torch.max(pred,-1)
  gt = batch_label
  accuracy_ = torch.eq(label,gt).to(device)
  accuracy_ = accuracy_.float()
  accuracy_ = torch.mean(accuracy_).to(device)
   
  
  return cost_, accuracy_



# testing in cross validation method
def testing(images_test, labels_test, pos_iter, device, n_steps, input_dims, model,loss):
  batch_label = torch.tensor(labels_test[pos_iter],device = device)
  _, batch_label = torch.max(batch_label,-1)
  batch_train = images_test[pos_iter]#[:, idx_permute]
  batch_train = batch_train.reshape([-1, n_steps, input_dims])
  batch_train = torch.tensor(batch_train)
  pred = model(batch_train).to(device)
  cost__ = loss(pred, batch_label).to(device)  # loss function
  _, label = pred.max(1)
  gt = batch_label
  accuracy__ = torch.eq(label, gt).to(device)
  accuracy__ = accuracy__.float()
  accuracy__ = torch.mean(accuracy__).to(device)
  
  return cost__,accuracy__
    
    
#batch_x_valid = batch_x_valid[:, idx_permute]
#batch_x_valid = batch_x_valid.reshape([-1, n_steps, input_dims])
#batch_x_valid = torch.tensor(batch_x_valid)
#current_acc = torch.tensor(0.0, device = device)




def train(images_train, labels_train, images_test, labels_test, n_steps, num_iters, device, input_dims, model,optimizer, loss):
  train_results = [] # saving ( loss_train, acc_train )
  test_results = []  # saving (loss_ test, acc_test )
  current_acc = torch.tensor(0.0, device = device) # update largest accuracy
  for pos_iter in range(num_iters):
    # training step 
    for (batch_train,batch_label) in zip(images_train[pos_iter], labels_train[pos_iter]):
      cost_,accuracy_ = train_iteration(batch_train, batch_label,device, n_steps, input_dims, model,optimizer,loss)
      train_results.append([cost_, accuracy_])
     
    # Testing step 
    cost__, accuracy__ = testing(images_test, labels_test,pos_iter, device, n_steps, input_dims,model, loss)
    test_results.append([cost__, accuracy__])
    # update best model 
    if ( accuracy__ > current_acc):
      current_acc = accuracy__
      best_model = copy.deepcopy(model)
    # print results
    print("********************************************** Step: ", pos_iter, " *************************************************")
    print('\n')
    print(' Test result: ', ' test loss: ', cost__, ' test acc: ', accuracy__)
    print('\n') 
    
  
  return best_model, train_results, test_results
  

################################################################## TRAINING ###################################################################

best_model, train_results, test_results =train(folds_images_train, folds_labels_train, folds_images_test, folds_labels_test, n_steps,num_iters,device, input_dims, DRNN, optimizer, loss)  
'''for pos_iter in range(num_iters): 
  for (batch_train,batch_label) in zip(folds_images_train[pos_iter], folds_labels_train[pos_iter]):
    batch_label = torch.tensor(batch_label,device = device)
    _, batch_label = torch.max(batch_label,-1)
    batch_train = batch_train[:, idx_permute]
    batch_train = batch_train.reshape([-1, n_steps, input_dims])
    batch_train = torch.tensor(batch_train)
    pred = DRNN(batch_train).to(device)

    optimizer.zero_grad()
    cost_ = nn.CrossEntropyLoss()(pred,batch_label).to(device) # loss function
    
    cost_.backward()
    optimizer.step()
    
    _,label = torch.max(pred,-1)
    gt = batch_label
    accuracy_ = torch.eq(label,gt).to(device)
    accuracy_ = accuracy_.float()
    accuracy_ = torch.mean(accuracy_).to(device)
    train_results.append([cost_, accuracy_])
  # testing 
  
  batch_label = torch.tensor(folds_labels_test[pos_iter],device = device)
  _, batch_label = torch.max(batch_label,-1)
  batch_train = folds_images_test[pos_iter][:, idx_permute]
  batch_train = batch_train.reshape([-1, n_steps, input_dims])
  batch_train = torch.tensor(batch_train)
  pred = DRNN(batch_train).to(device)
  cost__ = nn.CrossEntropyLoss()(pred, batch_label).to(device)  # loss function
  _, label = pred.max(1)
  gt = batch_label
  accuracy__ = torch.eq(label, gt).to(device)
  accuracy__ = accuracy_.float()
  accuracy__ = torch.mean(accuracy_).to(device)
  if ( accuracy__ > current_acc):
    current_acc = accuracy__
    best_model = copy.deepcopy(DRNN)
  test_results.append([cost__, accuracy__])
  print("********************************************** Step: ", pos_iter, " *************************************************")
  print('\n')
  print(' Test result: ', ' test loss: ', cost__, ' test acc: ', accuracy__)
  print('\n') '''

'''while step * batch_size < training_iters:
    batch_x, batch_y = mnist.test.next_batch(batch_size)
    batch_y = torch.tensor(batch_y,device=device)
    _,batch_y = torch.max(batch_y,-1)
    batch_x = batch_x[:, idx_permute]
    batch_x = batch_x.reshape([batch_size, n_steps, input_dims])
    batch_x = torch.tensor(batch_x)
    pred = DRNN(batch_x).to(device)

    optimizer.zero_grad()
    cost_ = nn.CrossEntropyLoss()(pred,batch_y).to(device) # loss function
    
    cost_.backward()
    optimizer.step()
    
    _,label = torch.max(pred,-1)
    gt = batch_y
    accuracy_ =  torch.eq(label,gt).to(device)
    accuracy_ = accuracy_.float()
    accuracy_ = torch.mean(accuracy_).to(device)
    train_results.append([cost_, accuracy_])
    
    if (step + 1) % display_step == 0:
        print("\n\n\n Iter " + str(step + 1) + ", Minibatch Loss: " + "{:.6f}".format(cost_) \
              + ", Training Accuracy: " + "{:.6f}".format(accuracy_))
    if (step + 1) % testing_step == 0:
        # validation performance
        # clean memory
        ''''''gc.collect()
    
        pred = DRNN(batch_x_valid).to(device)
        cost__ = nn.CrossEntropyLoss()(pred, batch_y_valid).to(device)  # loss function
        _, label = pred.max(1)
        gt = batch_y_valid
        accuracy__ = torch.eq(label, gt).to(device)
        accuracy__ = accuracy_.float()
        accuracy__ = torch.mean(accuracy_).to(device)

        validation_results.append([cost__, accuracy__])
        # test performance
        batch_x = mnist.test.images
        batch_y = mnist.test.labels
        batch_y = torch.tensor(batch_y,device=device)
        _, batch_y = torch.max(batch_y, -1)
        batch_x = batch_x[:, idx_permute]
        batch_x = batch_x.reshape([-1, n_steps, input_dims])
        batch_x = torch.tensor(batch_x)
        pred = DRNN(batch_x).to(device)
        cost_ = nn.CrossEntropyLoss()(pred, batch_y).to(device) # loss function
        _, label = pred.max(1)
        gt = batch_y
        accuracy_ = torch.eq(label, gt).to(device)
        accuracy_ = accuracy_.float()
        accuracy_ = torch.mean(accuracy_).to(device)
    
        test_results.append([cost_, accuracy_])
        print("\n\n\n ========> Validation Accuarcy: " + "{:.6f}".format(accuracy__)) ''''''
             # + ", Testing Accuarcy: " + "{:.6f}".format(accuracy_))
    
    step += 1
    
'''


##  ########################################################################################## PREDICT

torch.save(best_model.state_dict(),'modelDRNN.h5')
print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> parameters in model <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n');
for param_tensor in best_model.state_dict():
    print(param_tensor, "\t", best_model.state_dict()[param_tensor].size())


print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> TESTING <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<< \n');


data = mnist.train.images
data = data[:500]
#cddata = data[:, idx_permute]
data = data.reshape([-1, n_steps, input_dims])
data = torch.tensor(data)
gt = mnist.train.labels[:500]
gt = torch.tensor(gt)
_, gt = torch.max(gt, -1)

pred = best_model(data)
_,pred_label = pred.max(1)
 
acc = torch.eq(pred_label, gt).to(device)
acc = acc.float()
acc = torch.mean(acc).to(device)
print('acc: ', acc)


pred_label = np.resize(pred_label, (-1, 1))
gt_label = np.resize(gt, (-1,1))
loss = np.array(train_results)
loss = loss[:, 0]

plt.figure()
plt.plot(loss, '-')
plt.legend('loss values')
plt.title('loss of training')
plt.show()
#jijji
plt.figure()
plt.plot(gt_label, 'rx')
plt.plot(pred_label, 'bo')
plt.legend(('groundtruth', 'predict'))
plt.title('prediction test set')
plt.show()

