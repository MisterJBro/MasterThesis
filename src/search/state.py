import numpy as np
from copy import deepcopy


class State:
    """ State representation of the environment. """
    def __init__(self, env, done=False, rew=0.0):
        self.env = env
        self.done = done
        self.rew = rew

    def transition_inplace(self, action):
        _, self.rew, self.done, _ = self.env.step(action)
        self.rew = np.abs(self.rew)

    def transition(self, action):
        next_state = deepcopy(self)
        next_state.transition_inplace(action)
        return next_state

    def get_possible_actions(self):
        return self.env.available_actions()

    def is_terminal(self):
        return self.done

    def rollout(self, num_players):
        env = deepcopy(self.env)
        done = self.done
        ret = self.rew
        player = 0
        while not done:
            player = (player + 1) % num_players
            act = np.random.choice(env.available_actions())
            _, rew, done, _ = env.step(act)

            if player == 0:
                ret += np.abs(rew)
            else:
                ret -= np.abs(rew)
        return ret

    def __str__(self):
        return str(self.env)