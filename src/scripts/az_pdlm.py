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
    env = DiscreteActionWrapper(PendulumEnv())
    config = create_config({
        "env": env,
        "puct_c": 5.0,
        "az_iters": 1000,
        "az_eval_batch": 1,
        "num_trees": 1,
        "device": "cpu",
    })

    freeze_support()
    policy = PendulumPolicy(config)
    policy.load('checkpoints/policy_pdlm73_2.pt')
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
        ret_az = eval(env, get_action_az, render=True)
        rets_az.append(ret_az)
        ret_nn = eval(env, get_action_nn, render=False)
        rets_nn.append(ret_nn)
    print(f'AZ return: {np.mean(rets_az):.03f}')
    print(f'NN return: {np.mean(rets_nn):.03f}')
    env.close()
    az.close()
