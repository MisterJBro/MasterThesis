from copy import deepcopy
import time
import torch
import numpy as np
from multiprocessing import freeze_support
from src.networks.policy_pend import PendulumPolicy
from src.train.config import create_config
from src.search.alpha_zero import AlphaZero
from src.env.discretize_env import DiscreteActionWrapper
from src.env.pendulum import PendulumEnv
from src.search.state import State
from tqdm import tqdm


def eval(env, get_action, render=False):
    obs = env.reset()
    env.env.state = np.array([np.pi, 0.0])

    done = False
    ret = 0
    while not done:
        act = get_action(env, obs)
        obs, reward, done, _ = env.step(act)
        ret += reward

        if render:
            render_env = deepcopy(env)
            render_env.render()
    if render:
        render_env.close()
    return ret


if __name__ == '__main__':
    env = DiscreteActionWrapper(PendulumEnv(), n_bins=11)
    config = create_config({
        "env": env,
        "puct_c": 20.0,
        "train_iters": 100,
        "az_iters": 200,
        "az_eval_batch": 1,
        "num_cpus": 3,
        "num_envs": 15,
        "num_trees": 1,
        "device": "cpu",
        "pi_lr": 8e-4,
        "vf_lr": 5e-4,
        "sample_len": 500,
    })

    freeze_support()
    policy = PendulumPolicy(config)
    policy.load()
    az = AlphaZero(policy, config)

    def get_action_az(env, obs):
        az.update_policy(policy.state_dict())
        qvals = az.search(State(env, obs=obs))

        act = env.available_actions()[np.argmax(qvals)]
        return act

    def get_action_nn(env, obs):
        obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            dist = policy.get_dist(obs)
        act = dist.logits.argmax(-1)
        #act = dist.sample()

        return act.cpu().numpy()

    # Eval
    rets_az = []
    rets_nn = []
    for _ in tqdm(range(1)):
        ret_az = eval(env, get_action_az, render=False)
        rets_az.append(ret_az)
        #ret_nn = eval(env, get_action_nn, render=False)
        #rets_nn.append(ret_nn)
    print(f'AZ return: {np.mean(rets_az):.03f}')
    # 55.176
    #print(f'NN return: {np.mean(rets_nn):.03f}')
    env.close()
    az.close()
