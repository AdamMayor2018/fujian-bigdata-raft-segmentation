# -- coding: utf-8 --
# @Time : 2023/12/28 14:16
# @Author : caoxiang
# @File : utils.py
# @Software: PyCharm
# @Description:
import random
import torch
import numpy as np
import os
import time
from os.path import isfile
from os.path import join as opj


def fix_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True


def elapsed_time(start_time):
    return time.time() - start_time


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

