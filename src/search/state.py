import random
import numpy as np
from copy import deepcopy


class State:

    """ State representation of the environment. """
    def __init__(self, env, done=False, rew=0.0, obs=None):
        self.env = env
        self.done = done
        self.rew = rew
        self.obs = obs
        self.legal_act = self.env.legal_actions()

    def transition_inplace(self, action):
        self.obs, self.rew, self.done, _ = self.env.step(action)
        self.legal_act = self.env.legal_actions()

    def transition(self, action):
        next_state = deepcopy(self)
        next_state.transition_inplace(action)
        return next_state

    def get_possible_actions(self):
        return np.arange(self.env.size**2)[self.legal_act]

    def is_terminal(self):
        return self.done

    def rollout(self, num_players, gamma):
        env = deepcopy(self.env)
        done = self.done
        ret = 0
        player = 0
        while not done:
            player = (player + 1) % num_players
            act = random.choice(env.legal_actions())
            _, rew, done, _ = env.step(act)

            # Return
            if num_players == 2:
                if player == 0:
                    ret = rew + gamma*ret
                else:
                    ret = -rew + gamma*ret
            else:
                ret = rew + gamma*ret
        return ret

    def __str__(self):
        return str(self.env)


class ModelState:

    """ State representation of the environment. """
    def __init__(self, abs, rew=0.0, obs=None):
        self.abs = abs
        self.rew = rew
        self.obs = obs

    def __str__(self):
        return str(self.abs)