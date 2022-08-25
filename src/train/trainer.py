import numpy as np
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from copy import deepcopy

import gym
import time
import random
from src.env.envs import Envs
from src.networks.model import ValueEquivalenceModel
from src.search.model_search import plan
from src.networks.policy import ActorCriticPolicy
from src.train.log import Logger

from multiprocessing import freeze_support
from src.train.processer import post_processing
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
from tabulate import tabulate
from src.env.sample_batch import SampleBatch
from torch.utils.data import TensorDataset, DataLoader
from torch.distributions.kl import kl_divergence
import pathlib

PROJECT_PATH = pathlib.Path(__file__).parent.parent.parent.absolute().as_posix()


class Trainer(ABC):
    def __init__(self, config):
        # RNG seed
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.config = config
        self.device = config["device"]
        self.envs = Envs(config)
        self.policy = None
        self.writer = SummaryWriter(log_dir="../runs",comment=f'{config["env"]}_{config["num_samples"]}')
        self.log = Logger()

        print(tabulate([
            ['Environment', config["env"]],
            ['Obs shape', config["obs_dim"]],
            ['Actions num', config["num_acts"]],
            ['CPU count', config["num_cpus"]],
        ], colalign=("left", "right")))
        print()

    def train(self):
        for iter in range(self.config["train_iters"]):
            self.log.clear()
            self.log("Iter", iter)
            sample_batch = self.get_sample_batch()
            self.log.update(sample_batch.metrics)

            self.update(sample_batch)
            print(self.log)
            self.writer.add_scalar('Average return', self.log["avg ret"], iter)
            self.checkpoint()

    def checkpoint(self):
        if self.log["avg ret"] > self.log.best_metric:
            self.log.best_metric = self.log["avg ret"]
            self.best_model_path = f'{PROJECT_PATH}/checkpoints/policy_{self.config["env"]}_{self.__class__.__name__.lower()}_{self.log.best_metric:.0f}.pt'
            self.save(path=self.best_model_path)

    @abstractmethod
    def update(self, sample_batch):
        pass

    def get_sample_batch(self):
        sample_batch = SampleBatch(self.config)
        obs = self.envs.reset()

        for _ in range(self.config["sample_len"]):
            act, dist = self.get_action(obs)
            obs_next, rew, done = self.envs.step(act)

            sample_batch.append(obs, act, rew, done, dist)
            obs = obs_next

        sample_batch.set_last_obs(obs)
        sample_batch = post_processing(self.policy, sample_batch, self.config)
        return sample_batch

    def get_action(self, obs, envs=None, use_best=False):
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            dist = self.policy.get_dist(obs)
        if use_best:
            act = dist.logits.argmax(-1)
        else:
            act = dist.sample()

        return act.cpu().numpy(), dist.logits.cpu().numpy()

    def test(self, render=True):
        if isinstance(self.config["env"], str):
            env = gym.make(self.config["env"])
        else:
            env = self.config["env"]
        rews = []
        if render:
            input('Press any key to continue...')

        obs = env.reset()
        for _ in range(self.config["test_len"]):
            if render:
                deepcopy(env).render()
                time.sleep(0.1)
            act, _ = self.get_action(obs, envs=[deepcopy(env)], use_best=True)
            obs, rew, done, _ = env.step(act)
            rews.append(rew)

            if done:
                obs = env.reset()
                break
        print(f'Undiscounted return: {np.sum(rews)}')
        env.close()

    def save(self, path=None):
        if path is not None:
            self.policy.save(path=path)
        else:
            path = f'{PROJECT_PATH}/checkpoints/policy_{self.config["env"]}_{self.__class__.__name__.lower()}.pt'
            self.policy.save(path=path)

    def load(self, path=None):
        if path is not None:
            self.policy.load(path=path)
        else:
            path = f'{PROJECT_PATH}/checkpoints/policy_{self.config["env"]}_{self.__class__.__name__.lower()}.pt'
            self.policy.load(path=path)

    def __enter__(self):
        freeze_support()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.envs.close()
        self.writer.flush()
        self.writer.close()


class ModelTrainer:
    def __init__(self, config):
        # RNG seed
        seed = config["seed"]
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        self.config = config
        self.device = config["device"]
        self.envs = Envs(config)
        self.policy = ActorCriticPolicy(config)
        self.model = ValueEquivalenceModel(config)
        self.writer = SummaryWriter(comment=f'{config["env"]}_{config["num_samples"]}')

        print(tabulate([
            ['Environment', config["env"]],
            ['Obs shape', config["obs_dim"]],
            ['Actions num', config["num_acts"]],
            ['CPU count', config["num_cpus"]],
        ], colalign=("left", "right")))
        print()

    def train(self):
        for iter in range(self.config["train_iters"]):
            start = time.time()
            sample_batch = self.get_sample_batch()
            end = time.time()
            stats = sample_batch.statistics
            self.update(sample_batch)

            avg_ret = stats["mean_return"]
            max_ret = stats["max_return"]
            min_ret = stats["min_return"]
            print(f'Iteration: {iter}  Avg Ret: {np.round(avg_ret, 3)}  Max Ret: {np.round(max_ret, 3)}  Min Ret: {np.round(min_ret, 3)}')
            self.writer.add_scalar('Average return', avg_ret, iter)

    def get_sample_batch(self):
        sample_batch = SampleBatch(self.config)
        obs = self.envs.reset()

        for _ in range(self.config["sample_len"]):
            act = self.policy.get_action(obs)
            obs_next, rew, done = self.envs.step(act)

            sample_batch.append(obs, act, rew, done)
            obs = obs_next

        sample_batch.set_last_obs(obs)
        sample_batch = post_processing(self.policy, sample_batch, self.config)
        return sample_batch

    def update(self, sample_batch):
        data = sample_batch.to_tensor_dict()
        obs = data["obs"]
        ret = data["ret"]
        val = data["val"]

        # Time
        start = time.time()
        plan_targets = plan(self.policy, self.model, data, self.config)
        plan_actions = plan_targets.logits.argmax(-1)
        end = time.time()
        #print(f'Plan time: {end - start}')

        # Distill planning targets into policy
        trainset = TensorDataset(obs, plan_targets.logits)
        trainloader = DataLoader(trainset, batch_size=int(self.config["num_samples"]/10), shuffle=True)

        for i in range(20):
            for obs_batch, plan_target_batch in trainloader:
                self.policy.opt_policy.zero_grad()
                dist_batch = self.policy.get_dist(obs_batch)
                loss = kl_divergence(Categorical(logits=plan_target_batch), dist_batch).mean()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.policy.parameters(),  self.config["grad_clip"])
                self.policy.opt_policy.step()

        #act = data["act"]
        #act_onehot = to_onehot(act, self.config["num_acts"])
        #with torch.no_grad():
        #    act_model = self.model.dyn_linear(act_onehot)
        #    state = self.model.representation(obs)
        #    hidden, _ = self.model.dynamics(state, act_model)
        #    model_val = self.model.get_value(hidden)
        #    model_rew = self.model.get_reward(hidden)
        #q_val = model_rew + self.config["gamma"] * model_val

        adv = ret - val
        data["adv"] = adv

        # Policy and Value loss
        #self.policy.loss_gradient(data)
        self.policy.loss_value(data)

        # Get new logits for model loss
        with torch.no_grad():
            data["logits"] = self.policy.get_dist(obs).logits

        # Model loss
        self.model.loss(data)

    def test(self):
        env = gym.make(self.config["env"])
        rews = []
        input('Press any key to continue...')

        obs = env.reset()
        for _ in range(self.config["test_len"]):
            env.render()
            act = self.policy.get_action(obs)
            obs, rew, done, _ = env.step(act)
            rews.append(rew)
            if done:
                break
        print(f'Undiscounted return: {np.sum(rews)}')
        env.close()

    def __enter__(self):
        freeze_support()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.envs.close()
        self.writer.flush()
        self.writer.close()


