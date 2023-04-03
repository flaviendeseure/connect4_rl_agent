import numpy as np

from projet.agent.base_agent import Agent


class QLearner(Agent):
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

    def reset(self):
        self.q_values = {}
