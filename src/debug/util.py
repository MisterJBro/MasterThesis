import time
import random
import numpy as np
import torch

def measure_time_with_timer(fn, np_timer):
    """ Measures the time, which fn needs to be executed and saves it to np_timer"""
    start = time.time()
    res = fn()
    end = time.time()
    np_timer += end - start
    return res

def measure_time(fn):
    """ Measures the time, which fn needs to be executed and returns it"""
    start = time.time()
    res = fn()
    end = time.time()
    return res, end - start

def seed_all(seed):
    """ Seed random number generators using the given seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
