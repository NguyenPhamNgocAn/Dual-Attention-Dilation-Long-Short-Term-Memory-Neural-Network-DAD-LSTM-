import json
import os

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.externals import joblib
#  changing into modulescus if using modulescus module
from modules import Encoder, Decoder
from utils import numpy_to_tvar
import utils
from custom_types import TrainData
from constants import device


def preprocess_data(dat, col_names, scale) -> TrainData:
    # divide data into driving series(feats) and y history (targs) correspond to T-1 time steps
    proc_dat = scale.transform(dat)

    mask = np.ones(proc_dat.shape[1], dtype=bool)
    dat_cols = list(dat.columns)
    for col_name in col_names:
        mask[dat_cols.index(col_name)] = False

    feats = proc_dat[:, mask]
    targs = proc_dat[:, ~mask]

    return TrainData(feats, targs)


def predict(encoder, decoder, t_dat, batch_size: int, T: int) -> np.ndarray:
    print("shape 1 :", t_dat.feats.shape)
    print("shape 2 : ", t_dat.targs.shape)
    y_pred = np.zeros((t_dat.feats.shape[0] - T + 1, t_dat.targs.shape[1])) #!!!!!!!!!!!!!!!

    for y_i in range(0, len(y_pred), batch_size):
        y_slc = slice(y_i, y_i + batch_size)
        batch_idx = range(len(y_pred))[y_slc]
        b_len = len(batch_idx)
        X = np.zeros((b_len, T - 1, t_dat.feats.shape[1]))
        print("X shape : " , X.shape)
        y_history = np.zeros((b_len, T - 1, t_dat.targs.shape[1])) #!!!!!!!!!!!!!!!!!!!!!!!

        for b_i, b_idx in enumerate(batch_idx):
            idx = range(b_idx, b_idx + T - 1)

            X[b_i, :, :] = t_dat.feats[idx, :]
            y_history[b_i, :] = t_dat.targs[idx]

        y_history = numpy_to_tvar(y_history)
        # X has numpy type while encoder require a torch type ( since expressions in encoder using pytorch , we must change type )
        input_weighted, input_encoded = encoder(numpy_to_tvar(X))
        print("y_ history ", y_history.shape)
        print("encoder ",input_encoded.shape)
        print("y_slc ", y_slc)
        # .data is redundant since output values are the same
        # using with modulescus
        y_pred[y_slc] = decoder(input_encoded, y_history).cpu().data.numpy()
        # using with modules
        #y_pred[y_slc] = decoder(input_encoded, y_history).cpu().data.numpy()
    return y_pred




debug = False


save_plots = False
#  enc_kwargs = {"input_size": train_data.feats.shape[1], "hidden_size": encoder_hidden_size, "T": T}

print("device")
print(device)
with open(os.path.join("data", "enc_kwargs.json"), "r") as fi:
    enc_kwargs = json.load(fi)
enc = Encoder(**enc_kwargs).cuda(device)  # initialing class Encoder
# load hidden states in encoder of trained model
enc.load_state_dict(torch.load(os.path.join("data", "encoder.torch"), map_location=device))

with open(os.path.join("data", "dec_kwargs.json"), "r") as fi:
    dec_kwargs = json.load(fi)
# load hidden states in decoder of trained model
dec = Decoder(**dec_kwargs).cuda(device)   # initialing class Decoder

# from torch.save(model.encoder.state_dict(), os.path.join("data", "encoder.torch"))
# torch.save(model.decoder.state_dict(), os.path.join("data", "decoder.torch"))
dec.load_state_dict(torch.load(os.path.join("data", "decoder.torch"), map_location=device))

scaler = joblib.load(os.path.join("data", "scaler.pkl"))
raw_data = pd.read_csv(os.path.join("data", "nasdaq100_padding.csv"), nrows=100 if debug else None)
raw_data=raw_data.iloc[0:5000,:]
print("raw_data ",raw_data.shape)
targ_cols = ("NDX",)
data = preprocess_data(raw_data, targ_cols, scaler)
with open(os.path.join("data", "da_rnn_kwargs.json"), "r") as fi:
    da_rnn_kwargs = json.load(fi)
final_y_pred = predict(enc, dec, data, **da_rnn_kwargs)

plt.figure()
plt.plot(final_y_pred, label='Predicted')
plt.plot(data.targs[(da_rnn_kwargs["T"]-1):], label="True")
plt.legend(loc='upper left')

plt.show()
#plt.pause(2)
#plt.close()


#utils.save_or_show_plot("final_predicted_reloaded.png", save_plots)


