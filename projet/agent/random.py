from projet.agent.base_agent import Agent


class Random(Agent):
    def __init__(self,
                 action_space,
                 agent,
                 gamma=0.99,
                 eps_init=.5,
                 eps_min=1e-5,
                 eps_step=1 - 3,
                 name='Random'):
        super().__init__(action_space, agent, gamma, eps_init, eps_min, eps_step, name)

    def update(self, obs, action, reward, terminated, next_obs):
        pass

    def eps_greedy(self, obs):
        return self.action_space(self.agent).sample(mask=obs["action_mask"])

    def reset(self):
        pass
