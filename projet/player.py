import numpy as np


class Human:
    def __init__(self):
        pass

    def get_action(self, mask):
        action = ""
        available_actions = np.flatnonzero(mask["action_mask"])
        while not action.isdigit() or int(action) not in available_actions:
            action = input("Command: ")
        return np.int64(action)



class Agent:
    def __init__(self,
                 action_space,
                 agent,
                 gamma=0.99,
                 eps_init=.5,
                 eps_min=1e-5,
                 eps_step=1 - 3,
                 name="Agent"):
        
        self.action_space = action_space
        self.agent = agent
        self.gamma = gamma
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
        raise NotImplementedError
    
    def epsilon_decay(self):
        self.eps = max(self.eps - self.eps_step, self.eps_min)

    def reset(self):
        raise NotImplementedError


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
        
        self.Q[obs, action] += self.lr * (reward + self.gamma * self.Q[next_obs, next_action] - self.Q[obs, action])

        self.epsilon_decay()

    def reset(self):
        self.Q = {}



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
            return np.random.choice(np.flatnonzero(b == np.max(b))) # argmax with random tie-breaking
            #return np.argmax(b)
        
    def get_action(self, obs): 
        return self.eps_greedy(obs)
        
    def update(self, obs, action, reward, terminated, next_obs):
        self.current_episode_rewards.append(reward)
        self.current_episode_obs.append((obs, action))
        
        if terminated:
            self._end_of_episode_update()

    def _end_of_episode_update(self):
        current_episode_returns = rewardseq_to_returns(self.current_episode_rewards, self.gamma)
        seen = []
        for i, (state, action) in enumerate(self.current_episode_obs): 
            if (state, action) not in seen: 
                seen.append((state, action))
                return_value = current_episode_returns[i]
                
                n = self.number_of_values_in_estimate[state, action]
                self.values_estimates[state][action] = n / (n + 1) * self.values_estimates[state][action] + 1 / (n+1) * return_value
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
        self.number_of_values_in_estimate = defaultdict(lambda : 0)

