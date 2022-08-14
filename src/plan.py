import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from src.model import to_onehot


def max_search(state, hidden, curr_ret, curr_depth, all_acts, model, config):
    batch_size = hidden.shape[0]

    if curr_depth == config["max_search_depth"]:
        val = model.get_value(hidden)
        return curr_ret + (config["gamma"] ** curr_depth) * val

    max_ret = torch.zeros((batch_size,)).to(config["device"]) - float("inf")
    for act_idx, act in enumerate(all_acts):
        act = act.reshape(1, 1, -1).repeat(batch_size, 1, 1)
        next_hidden, next_state = model.dynamics(state, act)

        rew = model.get_reward(next_hidden)
        #dist = model.get_policy(next_hidden)

        next_ret = curr_ret + (config["gamma"] ** curr_depth) * rew
        curr_ret = max_search(next_state, next_hidden, next_ret, curr_depth + 1, all_acts, model, config)
        max_ret = torch.maximum(max_ret, curr_ret)
    return max_ret

def plan(policy, model, data, config):
    """ Plans on each observation of the data given, and returns action targets"""
    obs = data["obs"]
    act = data["act"]
    rew = data["rew"]
    done = data["done"]
    ret = data["ret"]
    val = data["val"]
    last_val = data["last_val"]
    sections = data["sections"]
    scalar_loss = nn.HuberLoss()
    act_onehot = to_onehot(act, config["num_acts"])

    # Get all actions by iterating from 0 to num_acts and then to_onehot and model.dyn_linear
    all_acts = []
    for i in range(config["num_acts"]):
        all_acts.append(to_onehot(torch.tensor([i]), config["num_acts"]))
    all_acts = torch.concat(all_acts).to(config["device"])
    with torch.no_grad():
        all_acts = model.dyn_linear(all_acts)

    config["max_search_depth"] = 4
    # Iterate over all observation
    all_qvals = []
    for (start, end) in sections:
        batch_size = end - start
        o = obs[start:end]
        state = model.representation(o)

        qvals = torch.zeros((batch_size, config["num_acts"]))
        for act_idx, a in enumerate(all_acts):
            with torch.no_grad():
                a = a.reshape(1, 1, -1).repeat(batch_size, 1, 1)
                hidden, next_state = model.dynamics(state, a)

                first_rew = model.get_reward(hidden)
                qvals[:, act_idx] = max_search(next_state, hidden, first_rew, 1, all_acts, model, config)
        all_qvals.append(qvals)
    all_qvals = torch.concat(all_qvals).to(config["device"])

    with torch.no_grad():
        logits = policy.get_dist(obs).logits

    dist_targets = Categorical(logits=logits + all_qvals*1.0)

    return dist_targets