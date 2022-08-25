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

@jit(nopython=True)
def gen_adv_estimation(rew, val, gamma, lam):
    len = rew.shape[0]

    adv_buf = np.zeros(len, dtype=np.float32)
    deltas = rew + gamma*val[1:] - val[:-1]

    for i in range(len):
        adv_buf[:len - i] += (lam*gamma)**i * deltas[i:]

    return adv_buf

@jit(nopython=True)
def calc_return(done, rew, gamma, last_val):
    ret = np.zeros((rew.shape))
    for b in range(rew.shape[0]):
        start = 0
        for t in range(rew.shape[1]):
            end = t + 1
            if done[b][t]:
                ret[b][start:end] = discount_cumsum(rew[b][start:end], gamma)
                start = end
            elif t == rew.shape[1] - 1:
                lv = last_val[b]
                # Episode was cut --> bootstrap
                ret[b][start:] = discount_cumsum(np.append(rew[b][start:], lv), gamma)[:-1]
    return ret

def calc_metrics(sample_batch):
    rew = sample_batch.rew
    done = sample_batch.done

    returns = []
    for b in range(rew.shape[0]):
        start = 0
        for t in range(rew.shape[1]):
            end = t + 1
            if done[b][t]:
                returns.append(np.sum(rew[b][start:end]))
                start = end

    if len(returns) == 0:
        returns = [np.sum(rew[b]) for b in range(rew.shape[0])]

    return {
        'avg ret': np.mean(returns),
        'max ret': np.max(returns),
        'min ret': np.min(returns),
    }

def post_processing(policy, sample_batch, config):
    # Value prediction
    buffer_shape = sample_batch.rew.shape
    obs = torch.as_tensor(sample_batch.obs)
    obs = obs.reshape(-1, obs.shape[-1]).to(policy.device)
    last_obs = torch.as_tensor(sample_batch.last_obs).to(policy.device)

    with torch.no_grad():
        val = policy.get_value(obs)
        val = val.reshape(buffer_shape)
        val = val.cpu().numpy()

        last_val = policy.get_value(last_obs)
        last_val = last_val.cpu().numpy()

    sample_batch.val = val
    sample_batch.last_val = last_val

    # Return calculation
    ret = calc_return(sample_batch.done, sample_batch.rew, config["gamma"], last_val)
    sample_batch.ret = ret

    # Calculate several metrics
    sample_batch.metrics = calc_metrics(sample_batch)

    return sample_batch
