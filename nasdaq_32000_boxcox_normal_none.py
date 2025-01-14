import typing
from typing import Tuple
import json
import os
import math
import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from DRNN_PYTORCH.classification_modelsPytorch import _contruct_cells
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import utils_1
from mdc  import Encoder, Decoder
from custom_types import DaRnnNet, TrainData, TrainConfig
from utils_1 import numpy_to_tvar
from constants_1 import device
import time
import decimal
import scipy.special as ss
torch.set_printoptions(precision=20)


start = time.time()
logger = utils_1.setup_log()
logger.info(f"Using computation device: {device}")

def MAPELoss(output, target):
    #print('values: ', torch.max(torch.abs((output-target))))
    #print('all:', torch.abs((output-target)/target))
    #print('count outier: ' , torch.sum(torch.abs((output-target)/target) >100.0 , axis =0 ))
    return torch.mean(torch.abs((output-target)/target))

def preprocess_data(dat, col_names) -> Tuple[TrainData, StandardScaler]:

    # fit data to get mean , var of data
    # scale just has responsible for containing mean , var of data , it does not normalized data
    dat = ss.boxcox(dat,0)
    scale = StandardScaler().fit(dat)
    # normalizing data
    proc_dat = scale.transform(dat)
    # make a list of companies's name such that one of them is removed to predict
    mask = np.ones(proc_dat.shape[1], dtype=bool)
    # dat is a dataframe , thus dat.columns take names of columns
    # and convert to list t
    dat_cols = list(dat.columns)
    # col_names : list of names that you wanna predict
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False
    # feats : values of T-1 time steps of companies in history
    # targs : values of target company
    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]
    print('proc: ')
    print(proc_dat)
    return TrainData(feats, targs), scale

# building a da_rnn ( still not computing )
def da_rnn(train_data: TrainData, n_targs: int, train_size:int, valid_size:int, hidden_structs_enc, dilations_enc, cells_enc,num_layer_enc, mode_enc,
        hidden_structs_dec, dilations_dec, cells_dec,num_layer_dec, mode_dec, num_layer_rlstm_enc, num_layer_rlstm_dec,enc_size=81
        ,encoder_hidden_size=64, decoder_hidden_size=64, T=10, learning_rate=0.01, batch_size=128):

    train_cfg = TrainConfig(T, train_size, valid_size, batch_size, nn.MSELoss())
    logger.info(f"Training size: {train_cfg.train_size:d}.")

    enc_kwargs = {"input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}
    # **enc_kwargs : a way to pass arguments into a function
    encoder = Encoder(**enc_kwargs, hidden_structs= hidden_structs_enc,dilations= dilations_enc,
            cells = cells_enc, num_layer= num_layer_enc,mode= mode_enc,num_layer_rlstm = num_layer_rlstm_enc).cuda(device) # specify arguments for class encoder #///////////////////////////////////////////////
    # json.dump : saving information of a dictionary to a json file
    with open(os.path.join("data/cuda_0", "enc_kwargs_1.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
            "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}
    decoder = Decoder(**dec_kwargs,enc_size= enc_size, hidden_structs= hidden_structs_dec,dilations= dilations_dec,cells= cells_dec, num_layer= num_layer_dec,mode= mode_dec, num_layer_rlstm = num_layer_rlstm_dec).cuda(device)  # specify arguments for class decoder #////////////////////////////////////////////// trnsfer from "to" into cuda 
    with open(os.path.join("data/cuda_0", "dec_kwargs_1.json"), "w") as fi:
        json.dump(dec_kwargs, fi, indent=4)

    encoder_optimizer = optim.Adam(
        params=[p for p in encoder.parameters() if p.requires_grad],
        lr=learning_rate)
    decoder_optimizer = optim.Adam(
        params=[p for p in decoder.parameters() if p.requires_grad],
        lr=learning_rate)
    da_rnn_net = DaRnnNet(encoder, decoder, encoder_optimizer, decoder_optimizer)

    return train_cfg, da_rnn_net


def train(net: DaRnnNet, train_data: TrainData, t_cfg: TrainConfig, n_epochs=10, save_plots=True):
    error = 999999999.0
    best_model = copy.deepcopy(net)
    iter_per_epoch = len(range(0, t_cfg.train_size, t_cfg.batch_size)[:-1])
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    test_error_mae = []
    test_error_rmse = []
    test_error_mape = []
    logger.info(f"Iterations per epoch: {len(range(0, t_cfg.train_size, t_cfg.batch_size)[:-1]):3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0

    for e_i in range(n_epochs):
        print("epoches : ", e_i )
        # permuting {0 -> t_cfg.train_size - t_cfg.T )
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
        #print("perm_idx : " , perm_idx )

        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size)[:-1]:
            # in a dataframe that each company has collected values  of long period time , choosing random batch_size
            # points and make a time series with length = T for each one

            batch_idx = perm_idx[t_i:(t_i + t_cfg.batch_size)]
            #print( " batch_idx_shape: " , batch_idx.shape )
            #print( " batch_idx : " , batch_idx)

            # feats : values of T-1 time steps of companies ( without target company )
            # y_history : previous values of T-1 time steps of target company
            # y_target :value at Tth time step of target company
            feats, y_history, y_target = prep_train_data(batch_idx, t_cfg, train_data)
            #print("feats : ",feats.shape)
            #print(" y_history : " , y_history.shape )
            #print(" y_target :  " , y_target.shape )
            #print("type : " , type(feats ) )
            
            loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)
            # computing loss for each batch in each epoch
            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            # if (j / t_cfg.batch_size) % 50 == 0:
            #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / t_cfg.batch_size:, loss)
            n_iter += 1

            # reduce 10% after each 10000 steps
            adjust_learning_rate(net, n_iter)
            print( "batch size next ", t_i )


        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

        if e_i % 1 == 0:# 10->1
            '''y_valid_pred = predict(net, train_data,
                                  t_cfg.train_size, t_cfg.valid_size, t_cfg.batch_size, t_cfg.T, mode ="VALID"
                                  )'''

            final_y_pred = predict(net, train_data, t_cfg.train_size,t_cfg.valid_size, t_cfg.batch_size, t_cfg.T, mode = "TEST")
            error_mae = nn.L1Loss()(numpy_to_tvar(final_y_pred),numpy_to_tvar(train_data.targs[t_cfg.train_size+t_cfg.valid_size:])).cpu()
            error_mape = MAPELoss(numpy_to_tvar(final_y_pred),numpy_to_tvar(train_data.targs[t_cfg.train_size+t_cfg.valid_size:])).cpu()
            error_mse = nn.MSELoss()(numpy_to_tvar(final_y_pred),numpy_to_tvar(train_data.targs[t_cfg.train_size+t_cfg.valid_size:])).cpu()
            error_rmse = torch.sqrt(error_mse)
            # TODO: make this MSE and make it work for multiple inputs
            #print('y_pred:', len(y_valid_pred))
            #print('train:', len(train_data.targs[t_cfg.train_size:t_cfg.train_size+t_cfg.valid_size]))
            #val_loss = t_cfg.loss_func(numpy_to_tvar(y_valid_pred),numpy_to_tvar(train_data.targs[t_cfg.train_size:t_cfg.train_size+t_cfg.valid_size]))
            #valid_error[int(e_i/10)] = val_loss
            test_error_mae.append(error_mae)
            test_error_rmse.append(error_rmse)
            test_error_mape.append(error_mape)
            # y_valid_pred - train_data.targs[t_cfg.train_size:t_cfg.train_size+t_cfg.valid_size] # compute this for logger purpose
            
            logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, test loss mae: {test_error_mae[-1]}, test loss rmse: {test_error_rmse[-1]}, test loss mape: {test_error_mape[-1]}.")
            y_train_pred = predict(net, train_data,
                                   t_cfg.train_size, t_cfg.valid_size, t_cfg.batch_size, t_cfg.T,
                                   mode ="TRAIN")


            error_train= t_cfg.loss_func(numpy_to_tvar(y_train_pred),numpy_to_tvar(train_data.targs[t_cfg.T-1:t_cfg.train_size]))
            if(error_train < error):
                error = error_train
                best_model = copy.deepcopy(net)
            '''if( test_error_mae[-1] <0.05 and test_error_rmse[-1] < 0.05 and test_error_mape[-1] < 0.05):
                break
            if(e_i > n_epochs):
                break
            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targs[:t_cfg.train_size])+len(train_data.targs[t_cfg.train_size:])), train_data.targs[:],
                     label="True")
            plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred,
                     label='Predicted - Train')
            plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs[:]) + 1), final_y_pred, label='Predicted - test')
            plt.legend(loc='upper left')
            plt.savefig('result_per_epoch_cuda0_' + str(e_i)+'.png')
            '''
            print('error train: ')
            print(error_train)
    # test error
    print('mae: ')
    print(test_error_mae)
    print('rmse: ')
    print(test_error_rmse)
    print('mape: ')
    print(test_error_mape)
    print('error train: ')
    print(error)

    return iter_losses, epoch_losses, best_model


def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    # feats : train_data.feats.shape[1]) driving series at T-1 time steps
    # y_history :
    feats = np.zeros((len(batch_idx), t_cfg.T, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):  # ( stt , val )
        b_slc = slice(b_idx, b_idx + t_cfg.T)
        y_slc = slice(b_idx, b_idx + t_cfg.T-1)
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_history[b_i, :] = train_data.targs[y_slc]

    return feats, y_history, y_target



def adjust_learning_rate(net: DaRnnNet, n_iter: int):
    # TODO: Where did this Learning Rate adjustment schedule come from?
    # Should be modified to use Cosine Annealing with warm restarts https://www.jeremyjordan.me/nn-learning-rate/
    if n_iter % 10000 == 0 and n_iter > 0:
        for enc_params, dec_params in zip(net.enc_opt.param_groups, net.dec_opt.param_groups):
            enc_params['lr'] = enc_params['lr'] * 0.9
            dec_params['lr'] = dec_params['lr'] * 0.9


def train_iteration(t_net: DaRnnNet, loss_func: typing.Callable, X, y_history, y_target):
    # parameter.grad= 0
    # if there are more than one function computed derivative the same variable , then that var is added cumulatively

    # setting all para with required_grad = True equal to 0
    t_net.enc_opt.zero_grad()
    t_net.dec_opt.zero_grad()
    # compute l
    input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
    # using with modulescus
    #y_pred = t_net.decoder(input_encoded,input_weighted, numpy_to_tvar(y_history))
    # using with modules
    y_pred = t_net.decoder(input_encoded, numpy_to_tvar(y_history))
    y_true = numpy_to_tvar(y_target)
    loss = loss_func(y_pred, y_true)
    # derivative loss function by variables with required_grad = True
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item() # return scalar



def predict(t_net: DaRnnNet, t_dat: TrainData, train_size: int, valid_size:int, batch_size: int, T: int, mode = "TEST"): # on_train : whether this step spends for training or not ?
    out_size = t_dat.targs.shape[1]  # size of output =  1
    if mode == "TRAIN": # if this step is training
        y_pred = np.zeros((train_size - T + 1, out_size)) # in training step , we use train_size - T + 1 time series so we get thesame size of output
    elif mode == "TEST":
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size-valid_size, out_size)) # the rest of dataset
    elif mode =="VALID":
        y_pred = np.zeros((valid_size,out_size))
    else:
        raise('invalid keyword !')
    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if mode =="TRAIN":
                idx = range(b_idx, b_idx + T)
                ydx = range(b_idx, b_idx + T - 1)
            elif mode == "VALID" :
                idx = range(b_idx + train_size - T, b_idx + train_size)
                ydx = range(b_idx + train_size - T, b_idx + train_size - 1)
            else:
                idx = range(b_idx + train_size +valid_size- T, b_idx + train_size +valid_size)
                ydx = range(b_idx + train_size +valid_size- T, b_idx + train_size +valid_size-1)
            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[ydx]

        y_history = numpy_to_tvar(y_history)
        input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
        # using with modulescus
        #y_pred[y_slc] = t_net.decoder(input_encoded,input_weighted, y_history).cpu().data.numpy()
        # using with modules
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()  # /////////////////////////////////////////////// trannsfer from cpu to cuda
    return y_pred

########################################## READ AND PROCESS DATA ####################################
save_plots = False
debug = False
# NASDAQ: 
# SML2010: sml2010.csv
# VN: data_edit_inter_arima.csv, data_edit_pttt_arima.csv, data_edit_inter_arima.csv, data_cut.csv
# egg: egg.csv
raw_data = pd.read_csv(os.path.join("data","sml2010_half.csv"), nrows=100 if debug else None)
raw_data[raw_data<1e-03] = 1e-03
print("raw_data ( dataframe ) " , raw_data )
dat_cols =list(raw_data.columns)
print("name of companies ", dat_cols )
mask = np.ones(raw_data.shape[1], dtype=bool)
logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
# NASDAQ: NDX
# SML2010: Temperature_Comedor_Sensor
# VN: BSR
# egg: O1
targ_cols = ("Temperature_Comedor_Sensor",)
for col_name in targ_cols :
    mask[dat_cols.index(col_name)] = False


############################################# MODE ENCODER DECODER ##################################
mode_enc = "none"
mode_dec = "none"



############################################# DILATED DRNN ##########################################
input_dim_enc = 64
cell_type_enc = "RNN"
assert (cell_type_enc in ["RNN","LSTM","GRU"])
hidden_structs_enc = [64]*2
dilations_enc = [2,4]
assert (len(hidden_structs_enc) == len(dilations_enc))
cells_enc = _contruct_cells(input_dim_enc, hidden_structs_enc, cell_type_enc)
num_layer_enc = 2

input_dim_dec = 64
cell_type_dec = "RNN"
assert (cell_type_dec in ["RNN","LSTM","GRU"])
hidden_structs_dec = [64]*2
dilations_dec = [2,4]
assert (len(hidden_structs_dec) == len(dilations_dec))
cells_dec = _contruct_cells(input_dim_dec, hidden_structs_dec, cell_type_dec)
num_layer_dec = 2

########################################### RESIDUAL LSTM ##########################################
num_layer_rlstm_enc = 2
num_layer_rlstm_dec = 2



###################################################################################################


# .isnull() : check whether value is nan ( null ) , if it happen :True , else False
# .isnull().sum() : sum the number of null each columns
# .isnull().sum().sum() : sum the total of null value in dataframe
'''logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
targ_cols = ("NDX",)
for col_name in targ_cols :
    mask[dat_cols.index(col_name)] = False
'''

# scaler : containing information about variance and mean ( call it + "." to normalize data )
data, scaler = preprocess_data(raw_data, targ_cols)
# data.feats and data.targ have numpy.ndarray type
print("X shape  : " , data.feats.shape )

print(" y_history shape: ", data.targs.shape )

print("X type : ", type( data.feats))
print(" y_history type : " , type(data.targs))
# NASDAQ:  batch size: 128, enc size: 81 , train size:35100, valid size:2730
# SML2010: batch size: 128, enc size: 21, train size: 2688, valid size: 620
# VN:  batch size: 16, enc code: 43, train size: 498, valid: 115, lr : .001 ->.0001
# new one: bacth size: 640, valid: 122
da_rnn_kwargs = {"batch_size": 128, "T": 10,"decoder_hidden_size": 64, "encoder_hidden_size": 64, "enc_size":18}
config, model = da_rnn(data, n_targs=len(targ_cols),train_size=2000, valid_size =0, learning_rate=.001,hidden_structs_enc= hidden_structs_enc,
        dilations_enc= dilations_enc,cells_enc= cells_enc,num_layer_enc= num_layer_enc,num_layer_rlstm_enc = num_layer_rlstm_enc,
        mode_enc= mode_enc,hidden_structs_dec= hidden_structs_dec,dilations_dec = dilations_dec,cells_dec= cells_dec,
        num_layer_dec= num_layer_dec,num_layer_rlstm_dec=num_layer_rlstm_dec, mode_dec= mode_dec,  **da_rnn_kwargs)
iter_loss, epoch_loss,best_model = train(model, data, config, n_epochs=101, save_plots=False)
final_y_pred = predict(best_model, data, config.train_size,config.valid_size, config.batch_size, config.T, mode = "TEST")

end = time.time()

pred_inverted = scaler.inverse_transform(np.repeat(final_y_pred,19,axis = 1))
pred_inverted = ss.inv_boxcox(pred_inverted,0)

name = "sml_half"
#print('inverted: ')
#print(pred_inverted)
#print(' original: ')
train_inverted = raw_data.to_numpy()[config.train_size+config.valid_size:,0]
#print('shape truth: ',train_inverted.shape)

plt.figure()
plt.plot(pred_inverted[:,0],"o--", label = 'predicted values')
plt.plot(train_inverted,"o--", label = 'ground truth')
#plt.title("Prediction on NASDAQ100 data set")
plt.legend(loc='upper left')
plt.savefig('none_test_truth_origin_' + name + '.png')
plt.close()

# LOSS VALUES
# plot loss values after each batch through all epoch
# scaling  y_axis to log before plotting
plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss)
#plt.title(" loss value ( each step ) of training set " )
plt.savefig('none_loss_function_' + name + '.png')
#utils.save_or_show_plot("iter_loss.png", save_plots)
#utils.save_or_show_plot("loss_train ", save_plots)
plt.close()

# VALID VALUES
'''plt.figure()
plt.semilogy(range(len(valid_error)), valid_error)
plt.title(" valid values ( each 10 epoches ) of validation set " )
plt.savefig('valid_cuda0.png')
plt.close()
'''
plt.figure()
plt.plot(final_y_pred, label='Predicted values')
print("final y predict size : " , final_y_pred.shape)
plt.plot(data.targs[config.train_size+config.valid_size:], label="ground truth")
#plt.title("y true vs y predict ")
print("y true size : " , data.targs[config.train_size+config.valid_size:].shape )
plt.legend(loc='upper left')
plt.savefig('none_final_scaled_' + name + '.png')
#plt.show()
#plt.pause(2)
plt.close()
#utils.save_or_show_plot("final_predicted.png", save_plots)
# saving information to file : da_rnn_kwargs.json
with open(os.path.join("data/cuda_0", "none_da_rnn_kwargs.json"), "w") as fi:
    json.dump(da_rnn_kwargs, fi, indent=4)

joblib.dump(scaler, os.path.join("data/cuda_0", "none_scaler.pkl"))
# saving values of hidden states of encoder + decoder
torch.save(best_model.encoder.state_dict(), os.path.join("data/cuda_0", "none_encoder.torch"))
torch.save(best_model.decoder.state_dict(), os.path.join("data/cuda_0", "none_decoder.torch"))

#print('model_encoder: ')
#print(best_model.encoder.state_dict())
#print('model_decoder: ')
#print(best_model.decoder.state_dict())

print('loss iter: ')
print(iter_loss)

print('error: ')
error = nn.L1Loss()(numpy_to_tvar(final_y_pred),numpy_to_tvar(data.targs[config.train_size+config.valid_size:])).cpu()
error_mape = MAPELoss(numpy_to_tvar(final_y_pred),numpy_to_tvar(data.targs[config.train_size+config.valid_size:])).cpu()
error_mse = nn.MSELoss()(numpy_to_tvar(final_y_pred),numpy_to_tvar(data.targs[config.train_size+config.valid_size:])).cpu()
error_rmse = torch.sqrt(error_mse)
print('mae: ')
print(error)
print('rmse: ')
print(torch.sqrt(error_mse))
print('mape: ')
print(error_mape)
print('time complexity:')
print(end - start)
np.save('error_none_mae_' + name + '.npy',error)
np.save('error_none_mape_' + name + '.npy',error_mape)
np.save('error_none_rmse_' + name + '.npy',error_rmse)
np.save('none_time_' + name + '.npy', end- start)
np.save('none_predict_test_' + name + '.npy', pred_inverted[:,0])
