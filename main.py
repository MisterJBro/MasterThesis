import gym
import numpy as np
import ray

ray.init()

env = gym.make("CartPole-v1")
obs = env.reset()

for _ in range(500):
    env.render()

    act = env.action_space.sample()
    obs, rew, done, _ = env.step(act)

    if done:
        break
env.close()