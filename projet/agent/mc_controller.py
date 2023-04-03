from collections import defaultdict

import numpy as np


def rewardseq_to_returns(reward_list: list[float], gamma: float) -> list[float]:
    """
    Turns a list of rewards into the list of returns
    """
    G = 0
    returns_list = []
    for r in reward_list[::-1]:
        G = r + gamma * G
        returns_list.append(G)
    return returns_list[::-1]


class MCController:
    """
        Monte-Carlo control
    """

    def __init__(self,
                 action_space,
                 observation_space,
                 gamma=0.99,
                 eps_init=.5,
                 eps_min=1e-5,
                 eps_step=1e-3,
                 episodes_between_greedyfication=500,
                 name='MC Controller'
                 ):

        self.action_space = action_space
        self.observation_space = observation_space
        self.gamma = gamma
        self.name = name

        self.eps = eps_init
        self.eps_min = eps_min
        self.eps_step = eps_step

        self.episodes_between_greedyfication = episodes_between_greedyfication
        self.episode_counter = 0

        self.reset()

    def eps_greedy(self, obs, eps=None):
        if eps is None:
            eps = self.eps

        # Return a greedy action wrt the action values estimates.

        if np.random.random() < self.eps:
            return self.action_space.sample()
        else:
            b = self.q_values[obs]
            return np.random.choice(
                np.flatnonzero(b == np.max(b)))  # argmax with random tie-breaking
            # return np.argmax(b)

    def get_action(self, obs):
        return self.eps_greedy(obs)

    def update(self, obs, action, reward, terminated, next_obs):
        self.current_episode_rewards.append(reward)
        self.current_episode_obs.append((obs, action))

        if terminated:
            self._end_of_episode_update()

    def _end_of_episode_update(self):
        current_episode_returns = rewardseq_to_returns(self.current_episode_rewards,
                                                       self.gamma)
        seen = []
        for i, (state, action) in enumerate(self.current_episode_obs):
            if (state, action) not in seen:
                seen.append((state, action))
                return_value = current_episode_returns[i]

                n = self.number_of_values_in_estimate[state, action]
                self.values_estimates[state][action] = n / (n + 1) * \
                                                       self.values_estimates[state][
                                                           action] + 1 / (
                                                               n + 1) * return_value
                self.number_of_values_in_estimate[state, action] += 1

        self.current_episode_rewards = []
        self.current_episode_obs = []

        self.episode_counter += 1

        if self.episode_counter % self.episodes_between_greedyfication == 0:
            new_q_values = defaultdict(lambda: np.zeros(self.action_space.n))
            for state in self.values_estimates:
                new_q_values[state] = self.values_estimates[state]
            self.reset(q_values=new_q_values)
            self.epsilon_decay()

    def epsilon_decay(self):
        self.eps = max(self.eps - self.eps_step, self.eps_min)

    def reset(self, q_values=None):
        self.current_episode_rewards = []
        self.current_episode_obs = []

        if q_values is None:
            self.q_values = defaultdict(lambda: np.zeros(self.action_space.n))
        else:
            self.q_values = q_values

        self.values_estimates = defaultdict(lambda: np.zeros(self.action_space.n))
        self.number_of_values_in_estimate = defaultdict(lambda: 0)
