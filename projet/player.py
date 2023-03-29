import json
from collections import defaultdict

import numpy as np
import pickle


class Human:
    def __init__(self):
        pass

    def get_action(self, mask):
        action = ""
        available_actions = np.flatnonzero(mask["action_mask"])
        while not action.isdigit() or int(action) not in available_actions:
            action = input("Command: ")
        return np.int64(action)

class Random:
    def __init__(self, action_space, agent):
        self.action_space = action_space
        self.agent = agent

    def get_action(self, obs):
        return self.action_space(self.agent).sample(mask=obs["action_mask"])


class QLearner:
    """
        Stores the data and computes the observed returns.
    """

    def __init__(self,
                 action_space,
                 agent,
                 gamma=0.99,
                 lr=0.1,
                 eps_init=.5,
                 eps_min=1e-5,
                 eps_step=1 - 3,
                 name='Q-learning'):

        self.action_space = action_space
        self.gamma = gamma
        self.lr = lr
        self.agent = agent
        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_step = eps_step

        self.name = name

        self.reset()

    def eps_greedy(self, obs, eps=None):
        if eps is None:
            eps = self.eps
        if np.random.random() < self.eps:
            return self.action_space(self.agent).sample(mask=obs["action_mask"])
        else:
            b = self.q_values.get(str(obs["observation"]), np.zeros(7))
            return np.random.choice(
                np.flatnonzero(b == np.max(b)))  # argmax with random tie-breaking

    def get_action(self, obs):
        return self.eps_greedy(obs)

    def update(self, obs, action, reward, terminated, next_obs):
        next_obs = str(next_obs["observation"])
        obs = str(obs["observation"])
        estimate_value_at_next_state = (not terminated) * np.max(
            self.q_values.get(next_obs, np.zeros(7)))
        new_estimate = reward + self.gamma * estimate_value_at_next_state

        if obs not in self.q_values:
            self.q_values[obs] = np.zeros(7)

        self.q_values[obs][action] = (
                (1 - self.lr) * self.q_values[obs][action]
                + self.lr * new_estimate
        )

        self.epsilon_decay()

    def epsilon_decay(self):
        self.eps = max(self.eps - self.eps_step, self.eps_min)

    def reset(self):
        # self.q_values = defaultdict(lambda: np.zeros(7))
        self.q_values = {}


    @staticmethod
    def serialize_defaultdict(obj):
        if isinstance(obj, defaultdict):
            result = {}
            for k, v in obj.items():
                result[k] = v
            return result
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(f'Object of type {type(obj)} is not JSON serializable')

    @staticmethod
    def deserialize_defaultdict(d):
        result = defaultdict(lambda: 0)
        for k, v in d.items():
            if isinstance(v, list):
                result[k] = np.array(v)
            else:
                result[k] = v
        return result

    def save(self, path):
        pickle.dump(self.q_values, open(path, "wb"))

    def load(self, path):
        self.q_values = pickle.load(open(path, "rb"))
