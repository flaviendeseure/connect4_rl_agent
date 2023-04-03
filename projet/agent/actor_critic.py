import torch
from torch import nn

# NN layers
HIDDEN_SIZE = 128
# Discount factor
GAMMA = 0.99
# Learning rates
LR_ACTOR = 0.00001
LR_CRITIC = 0.0001
# Environment parameters
K_EPOCHS = 5  # update policy for K epochs in one PPO update
EPS_CLIP = 0.2  # clip parameter for PPO
RANDOM_SEED = 0  # set random seed if required (0 = no random seed)


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        # actor
        self.actor = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, action_dim),
            nn.Softmax(dim=-1),
        )
        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE),
            nn.Tanh(),
            nn.Linear(HIDDEN_SIZE, 1),
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        action_probs = self.actor(torch.transpose(state, 0, 1))
        # print(action_probs[0], torch.Tensor.mean(action_probs[0]))

        norm_dist = torch.distributions.MultivariateNormal(
            loc=action_probs[0], covariance_matrix=torch.diag(action_probs[0])
        )
        action = norm_dist.sample()

        action_logprob = norm_dist.log_prob(action)
        state_val = self.critic(torch.transpose(state, 0, 1))[0]

        action_clipped = torch.clip(
            (self.state_dim - 1) * action,
            min=torch.zeros(3, dtype=torch.int),
            max=torch.Tensor(
                [self.state_dim - 1, self.state_dim - 1, self.state_dim - 1]),
        ).int()

        return (
            action_clipped.detach(),
            action.detach(),
            action_logprob.detach(),
            state_val.detach(),
        )

    def evaluate(self, state, action):
        action_probs = []
        for i in range(state.shape[0]):
            action_probs.append(self.actor(torch.transpose(state[i], 0, 1))[0])
        action_probs = torch.stack(action_probs)

        # dist = Categorical(action_probs)
        norm_dist = torch.distributions.MultivariateNormal(
            loc=action_probs[0], covariance_matrix=torch.diag(action_probs[0])
        )
        action_logprobs = norm_dist.log_prob(action)
        dist_entropy = norm_dist.entropy()
        state_values = []
        for i in range(state.shape[0]):
            state_values.append(self.critic(torch.transpose(state[i], 0, 1))[0])
        state_values = torch.stack(state_values)

        return action_logprobs, state_values, dist_entropy
