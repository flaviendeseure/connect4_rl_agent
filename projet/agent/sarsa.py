import numpy as np

from projet.agent.base_agent import Agent


class Sarsa(Agent):
    def __init__(self,
                 action_space,
                 agent,
                 gamma=0.99,
                 lr=0.1,
                 eps_init=.5,
                 eps_min=1e-5,
                 eps_step=1 - 3,
                 name='Q-learning'):
        super().__init__(action_space, agent, gamma, eps_init, eps_min, eps_step, name)
        self.lr = lr

    def update(self, obs, action, reward, terminated, next_obs):
        next_action = self.eps_greedy(next_obs)

        next_obs = str(next_obs["observation"])
        obs = str(obs["observation"])

        if (obs, action) not in self.Q:
            self.Q[obs, action] = 0

        if (next_obs, next_action) not in self.Q:
            self.Q[next_obs, next_action] = 0

        self.Q[obs, action] += self.lr * (
                reward + self.gamma * self.Q[next_obs, next_action] - self.Q[
            obs, action]
        )

        self.epsilon_decay()

    def eps_greedy(self, obs, eps=None):
        if eps is None:
            eps = self.eps
        if np.random.random() < self.eps:
            return self.action_space(self.agent).sample(mask=obs["action_mask"])
        else:
            if self.Q == {}:
                return self.action_space(self.agent).sample(mask=obs["action_mask"])
            b = self.Q[str(obs["observation"]), :]
            return np.random.choice(
                np.flatnonzero(b == np.max(b))
            )

    def reset(self):
        self.Q = {}
