'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import logging
from pathlib2 import Path
from datetime import datetime


def getLogger(log_dir):
    logger = logging.getLogger('logger')
    logger.setLevel(logging.DEBUG)
    logging.basicConfig(format='%(message)s', level=logging.DEBUG)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    handler = logging.FileHandler(Path(log_dir,datetime.now().strftime('log_%Y_%m_%d_%H_%M_%S.log')))
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter('%(message)s'))
    logging.getLogger().addHandler(handler)
    return logger
