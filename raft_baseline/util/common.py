# -- coding: utf-8 --
# @Time : 2024/1/2 11:41
# @Author : caoxiang
# @File : common.py
# @Software: PyCharm
# @Description:通用工具函数
import random
import torch
import numpy as np
import os
import time


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
