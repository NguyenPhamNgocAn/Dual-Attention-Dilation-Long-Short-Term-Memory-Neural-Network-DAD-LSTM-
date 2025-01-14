import typing
from typing import Tuple
import json
import os

import torch
from torch import nn
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import copy
import utils
from modules_rlstm_version_2 import Encoder, Decoder
from custom_types import DaRnnNet, TrainData, TrainConfig
from utils import numpy_to_tvar
from constants import device
import time
import faulthandler 
logger = utils.setup_log()
logger.info(f"Using computation device: {device}")
faulthandler.enable()

start = time.time()
def preprocess_data(dat, col_names) -> Tuple[TrainData, StandardScaler]:

    # fit data to get mean , var of data
    # scale just has responsible for containing mean , var of data , it does not normalized data
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

    return TrainData(feats, targs), scale

# building a da_rnn ( still not computing )
def da_rnn(train_data: TrainData, n_targs: int,hidden_LSTM,hidden_dense_1, hidden_dense_2, input_dim_rlsrm,encoder_hidden_size=64, decoder_hidden_size=64, T=10, learning_rate=0.01, batch_size=128):
    train_cfg = TrainConfig(T, int(train_data.feats.shape[0] * 0.7), batch_size, nn.MSELoss())
    logger.info(f"Training size: {train_cfg.train_size:d}.")

    enc_kwargs = {"input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}
    # **enc_kwargs : a way to pass arguments into a function
    encoder = Encoder(**enc_kwargs) # specify arguments for class encoder #///////////////////////////////////////////////
    # json.dump : saving information of a dictionary to a json file
    with open(os.path.join("data", "enc_kwargs.json"), "w") as fi:
        json.dump(enc_kwargs, fi, indent=4)

    dec_kwargs = {"encoder_hidden_size": encoder_hidden_size,
            "decoder_hidden_size": decoder_hidden_size, "T": T, "out_feats": n_targs}
    decoder = Decoder(**dec_kwargs, hidden_LSTM =hidden_LSTM,hidden_dense_1= hidden_dense_1, hidden_dense_2= hidden_dense_2, input_dim_rlstm=input_dim_rlstm)  # specify arguments for class decoder #////////////////////////////////////////////// trnsfer from "to" into cuda 
    with open(os.path.join("data", "dec_kwargs.json"), "w") as fi:
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
    iter_per_epoch = int(np.ceil(t_cfg.train_size * 1. / t_cfg.batch_size))
    iter_losses = np.zeros(n_epochs * iter_per_epoch)
    epoch_losses = np.zeros(n_epochs)
    logger.info(f"Iterations per epoch: {t_cfg.train_size * 1. / t_cfg.batch_size:3.3f} ~ {iter_per_epoch:d}.")

    n_iter = 0
    error = 999999999.0
    best_model = copy.copy(net)
    for e_i in range(n_epochs):
        print("epoches : ", e_i )
        # permuting {0 -> t_cfg.train_size - t_cfg.T )
        perm_idx = np.random.permutation(t_cfg.train_size - t_cfg.T)
        #print("perm_idx : " , perm_idx )

        for t_i in range(0, t_cfg.train_size, t_cfg.batch_size):
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

            print( "batch_index: ", t_i )

            loss = train_iteration(net, t_cfg.loss_func, feats, y_history, y_target)
            # computing loss for each batch in each epoch
            iter_losses[e_i * iter_per_epoch + t_i // t_cfg.batch_size] = loss
            # if (j / t_cfg.batch_size) % 50 == 0:
            #    self.logger.info("Epoch %d, Batch %d: loss = %3.3f.", i, j / t_cfg.batch_size, loss)
            n_iter += 1

            # reduce 10% after each 10000 steps
            adjust_learning_rate(net, n_iter)

        epoch_losses[e_i] = np.mean(iter_losses[range(e_i * iter_per_epoch, (e_i + 1) * iter_per_epoch)])

        if e_i % 10 == 0:
            y_test_pred = predict(net, train_data,
                                  t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                  on_train=False)
            # TODO: make this MSE and make it work for multiple inputs
            val_loss = y_test_pred - train_data.targs[t_cfg.train_size:] # compute this for logger purpose
            logger.info(f"Epoch {e_i:d}, train loss: {epoch_losses[e_i]:3.3f}, val loss: {np.mean(np.abs(val_loss))}.")
            if ( error > np.mean(np.abs(val_loss)) ):
                error = np.mean(np.abs(val_loss))
                best_model = copy.deepcopy(net)
            y_train_pred = predict(net, train_data,
                                   t_cfg.train_size, t_cfg.batch_size, t_cfg.T,
                                   on_train=True)

            plt.figure()
            plt.plot(range(1, 1 + len(train_data.targs)), train_data.targs,
                     label="True")
            plt.plot(range(t_cfg.T, len(y_train_pred) + t_cfg.T), y_train_pred,
                     label='Predicted - Train')
            plt.plot(range(t_cfg.T + len(y_train_pred), len(train_data.targs) + 1), y_test_pred,
                     label='Predicted - Test')
            plt.legend(loc='upper left')
            plt.savefig('result_per_epoch_rlstm_version_2_' + str(e_i)+'.png')
            plt.show()
            #plt.pause(2)
            plt.close()
            #utils.save_or_show_plot(f"pred_{0}.png".format(e_i),save_plots)

    return iter_losses, epoch_losses, best_model

def prep_train_data(batch_idx: np.ndarray, t_cfg: TrainConfig, train_data: TrainData):
    # feats : train_data.feats.shape[1]) driving series at T-1 time steps
    # y_history :
    feats = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.feats.shape[1]))
    y_history = np.zeros((len(batch_idx), t_cfg.T - 1, train_data.targs.shape[1]))
    y_target = train_data.targs[batch_idx + t_cfg.T]

    for b_i, b_idx in enumerate(batch_idx):  # ( stt , val )
        b_slc = slice(b_idx, b_idx + t_cfg.T - 1)
        feats[b_i, :, :] = train_data.feats[b_slc, :]
        y_history[b_i, :] = train_data.targs[b_slc]

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
    y_pred = y_pred.cpu()
    y_true = y_true.cpu() 
    loss = loss_func(y_pred, y_true)
    # derivative loss function by variables with required_grad = True
    loss.backward()

    t_net.enc_opt.step()
    t_net.dec_opt.step()

    return loss.item() # return scalar

def predict(t_net: DaRnnNet, t_dat: TrainData, train_size: int, batch_size: int, T: int, on_train=False): # on_train : whether this step spends for training or not ?
    out_size = t_dat.targs.shape[1]  # size of output =  1
    if on_train: # if this step is training
        y_pred = np.zeros((train_size - T + 1, out_size)) # in training step , we use train_size - T + 1 time series so we get thesame size of output
    else:
        y_pred = np.zeros((t_dat.feats.shape[0] - train_size, out_size)) # the rest of dataset

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1]))

        for b_i, b_idx in enumerate(batch_idx):
            if on_train:
                idx = range(b_idx, b_idx + T - 1)
            else:
                idx = range(b_idx + train_size - T, b_idx + train_size - 1) # bị trùng lập ở batch_size đầu tiên

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        input_weighted, input_encoded = t_net.encoder(numpy_to_tvar(X))
        # using with modulescus
        #y_pred[y_slc] = t_net.decoder(input_encoded,input_weighted, y_history).cpu().data.numpy()
        # using with modules
        y_pred[y_slc] = t_net.decoder(input_encoded, y_history).cpu().data.numpy()  #quen roi !!!! /////////////////////////////////////////////// trannsfer from cpu to cuda
    return y_pred


save_plots = False
debug = False

raw_data = pd.read_csv(os.path.join("data", "nasdaq100_padding.csv"), nrows=100 if debug else None)

print("raw_data ( dataframe ) " , raw_data )
dat_cols =list(raw_data.columns)
print("name of companies ", dat_cols )
mask = np.ones(raw_data.shape[1], dtype=bool)
input_dim_rlstm = 64
hidden_LSTM = 100
hidden_dense_1 = 10
hidden_dense_2 = 1
num_layer = 2 # tuong trung thoi chu khong co dung
# .isnull() : check whether value is nan ( null ) , if it happen :True , else False
# .isnull().sum() : sum the number of null each columns
# .isnull().sum().sum() : sum the total of null value in dataframe
logger.info(f"Shape of data: {raw_data.shape}.\nMissing in data: {raw_data.isnull().sum().sum()}.")
targ_cols = ("NDX",)
for col_name in targ_cols :
    mask[dat_cols.index(col_name)] = False

#print(mask)



#print("length_ compaines : " , len(mask) )



# scaler : containing information about variance and mean ( call it + "." to normalize data )
data, scaler = preprocess_data(raw_data, targ_cols)
# data.feats and data.targ have numpy.ndarray type
print("X shape  : " , data.feats.shape )

print(" y_history shape: ", data.targs.shape )

print("X type : ", type( data.feats))
print(" y_history type : " , type(data.targs))
da_rnn_kwargs = {"batch_size": 128, "T": 10}
config, model = da_rnn(data, n_targs=len(targ_cols), learning_rate=.001,hidden_LSTM =hidden_LSTM,hidden_dense_1= hidden_dense_1, hidden_dense_2= hidden_dense_2, input_dim_rlsrm=input_dim_rlstm, **da_rnn_kwargs)
iter_loss, epoch_loss, best_model = train(model, data, config, n_epochs=100, save_plots=False)
final_y_pred = predict(best_model, data, config.train_size, config.batch_size, config.T)

end = time.time()
# plot loss values after each batch through all epoch
# scaling  y_axis to log before plotting
plt.figure()
plt.semilogy(range(len(iter_loss)), iter_loss)
plt.title(" loss value ( each step ) of training set " )
plt.savefig('loss_function_rlstm_version_2.png')
#utils.save_or_show_plot("iter_loss.png", save_plots)
#utils.save_or_show_plot("loss_train ", save_plots)
'''
plt.figure()
# loss for each epoch
plt.semilogy(range(len(epoch_loss)), epoch_loss )
plt.axis((0,len(epoch_loss),-1,1))
plt.show(block=False)
#plt.pause(2)
plt.close()
plt.show()
'''
plt.figure()        
plt.plot(final_y_pred, label='Predicted')
print("final y predict size : " , final_y_pred.shape)
plt.plot(data.targs[config.train_size:], label="True")
plt.title("y true vs y predict ")
print("y true size : " , data.targs[config.train_size:].shape )
plt.legend(loc='upper left')
plt.savefig('final_rlstm_version_2.png')
plt.show()
#plt.pause(2)
plt.close()
#utils.save_or_show_plot("final_predicted.png", save_plots)
# saving information to file : da_rnn_kwargs.json
with open(os.path.join("data", "da_rnn_kwargs.json"), "w") as fi:
    json.dump(da_rnn_kwargs, fi, indent=4)

joblib.dump(scaler, os.path.join("data", "scaler.pkl"))
# saving values of hidden states of encoder + decoder
torch.save(best_model.encoder.state_dict(), os.path.join("data", "encoder.torch"))
torch.save(best_model.decoder.state_dict(), os.path.join("data", "decoder.torch"))

print('model_encoder: ')
print(best_model.encoder.state_dict())
print('model_decoder: ')
print(best_model.decoder.state_dict())

print('loss iter: ')
print(iter_loss)

print('error: ')
err = final_y_pred - data.targs[config.train_size:]
err =np.mean(np.abs(err))
print(err)
np.save('error_rlstm_v_2.npy', err)
np.save('time_rlstm_v_2.npy', end-start)
