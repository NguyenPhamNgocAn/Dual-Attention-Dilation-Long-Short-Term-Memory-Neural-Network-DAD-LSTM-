import collections
import typing

import numpy as np

# if this is not inherit from typing.NamedTuple then can't declare classes like this
class TrainConfig(typing.NamedTuple):
  T: int
  train_size: int  # from original data we take k %
  valid_size: int
  batch_size: int
  loss_func: typing.Callable


class TrainData(typing.NamedTuple):
  feats: np.ndarray  # feats : X
  targs: np.ndarray  # targs : data of predict


DaRnnNet = collections.namedtuple("DaRnnNet", ["encoder", "decoder", "enc_opt", "dec_opt"])

