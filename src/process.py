import statistics
import torch
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import scipy.signal

def discount_cumsum2(x, discount):
    """from https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/vpg/core.py under MIT License"""
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

@jit(nopython=True)
def discount_cumsum(rew, gamma):
    len = rew.shape[0]
    ret = np.zeros(len, dtype=np.float32)
    for k in range(len):
        ret[:len - k] += (gamma**k)*rew[k:]
    return ret

def calc_return(done, rew, gamma, last_val):
    ret = np.zeros((rew.shape))
    for b in range(rew.shape[0]):
        start = 0
        end = 0
        for t in range(rew.shape[1]):
            if done[b][t]:
                ret[b][start:end] = discount_cumsum(rew[b][start:end], gamma)
                start = end
            elif t == rew.shape[1] - 1:
                # Episode was cut --> bootstrap
                ret[b][start:] = discount_cumsum(np.append(rew[b][start:], last_val[b]), gamma)[:-1]
            else:
                end += 1
    return ret

def calc_statistics(sample_batch):
    rew = sample_batch.rew
    done = sample_batch.done

    mean_returns = []
    for b in range(rew.shape[0]):
        start = 0
        end = 0
        for t in range(rew.shape[1]):
            if done[b][t]:
                mean_returns.append(np.sum(rew[b][start:end]))
                start = end
            elif t == rew.shape[1] - 1:
                mean_returns.append(np.sum(rew[b][start:end]))
            else:
                end += 1
    mean_return = np.mean(mean_returns)
    return {
        'mean_return': mean_return
    }

def post_processing(policy, sample_batch, config):
    # Value prediction
    buffer_shape = sample_batch.rew.shape
    obs = torch.tensor(sample_batch.obs)
    obs = obs.reshape(-1, obs.shape[-1])
    last_obs = torch.tensor(sample_batch.last_obs)

    with torch.no_grad():
        val = policy.get_value(obs)
        val = val.reshape(buffer_shape)
        val = val.cpu().numpy()

        last_val = policy.get_value(last_obs)
        last_val = last_val.cpu().numpy()

    sample_batch.val = val

    # Return calculation
    ret = calc_return(sample_batch.done, sample_batch.rew, config["gamma"], last_val)
    sample_batch.ret = ret

    # Calculate several statistics
    sample_batch.statistics = calc_statistics(sample_batch)

    return sample_batch
