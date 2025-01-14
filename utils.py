'''
Reference:
  https://realpython.com/python-logging/
'''

import logging
import os

import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable

from constants import device

# using for display information on console such as debug , error , warning...
# VOC = Visual Object Classes (VOC)
def setup_log(tag='VOC_TOPICS'): 
  # create logger
  logger = logging.getLogger(tag)
  # logger.handlers = []
  logger.propagate = False  # logging messages are not passed to the handlers of ancestor loggers
  logger.setLevel(logging.DEBUG) # set level for logger => log ALL
  # create console handler and set level to debug
  ch = logging.StreamHandler()
  ch.setLevel(logging.DEBUG)
  # create formatter
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  # add formatter to ch
  ch.setFormatter(formatter)
  # add ch to logger
  # logger.handlers = []
  logger.addHandler(ch)
  return logger


def save_or_show_plot(file_nm: str, save):
  if save:
    plt.savefig(os.path.join(os.path.dirname(__file__), "plots", file_nm))
  else:
    plt.show()


def numpy_to_tvar(x):
  # required grad is False if not declare
  return Variable(torch.from_numpy(x).type(torch.FloatTensor)).to(device)


import numpy as np
x = np.array([1, 2, 3])
print(x)
print(torch.from_numpy(x))


