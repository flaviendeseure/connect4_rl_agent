import pettingzoo
import torch
from torch import nn

from projet.agent.base_agent import Agent

import numpy as np  
import torch.optim as optim

HIDDEN_SIZE = 128


class ActorNetwork(nn.Module):
    def __init__(self, obs_size: int, action_size: int) -> None:
        super().__init__()
        self.conv1_layer = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
        self.relu1 = nn.ReLU(),
        self.conv2 =nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
        self.relu2 = nn.ReLU(),
        self.flatten = nn.Flatten(),
        self.fc1_layer: nn.Linear = nn.Linear(obs_size, HIDDEN_SIZE)
        self.fc2_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.mu_out_layer: nn.Sequential = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_size), nn.Sigmoid()
        )
        self.sigma_out_layer: nn.Sequential = nn.Sequential(
            nn.Linear(HIDDEN_SIZE, action_size), nn.Softplus()
        )

    def forward(
        self, input_x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.distributions.MultivariateNormal]:
        input_x = self.conv1_layer(input_x)
        input_x = self.relu1(input_x)
        input_x = self.conv2(input_x)
        input_x = self.relu2(input_x)
        input_x = self.flatten(input_x)
        input_x = self.fc1_layer(input_x)
        input_x = self.fc2_layer(input_x)
        output_mu: torch.Tensor = self.mu_out_layer(input_x)
        sigma_diag: torch.Tensor = self.sigma_out_layer(input_x)
        norm_dist = torch.distributions.MultivariateNormal(
            loc=output_mu, covariance_matrix=torch.diag(sigma_diag)
        )
        sample_x = norm_dist.sample()
        return sample_x.detach(), norm_dist


class CriticNetwork(nn.Module):
    def __init__(self, obs_size: int) -> None:
        super().__init__()
        self.conv1_layer = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0),
        self.relu1 = nn.ReLU(),
        self.conv2 =nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0),
        self.relu2 = nn.ReLU(),
        self.flatten = nn.Flatten(),
        self.fc1_layer: nn.Linear = nn.Linear(obs_size, HIDDEN_SIZE)
        self.fc2_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.critic_out_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, 1)
        self.sigmoid: nn.Sigmoid = nn.Sigmoid()

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        input_x = self.conv1_layer(input_x)
        input_x = self.relu1(input_x)
        input_x = self.conv2(input_x)
        input_x = self.relu2(input_x)
        input_x = self.flatten(input_x)
        input_x = self.fc1_layer(input_x)
        input_x = self.sigmoid(input_x)
        input_x = self.fc2_layer(input_x)
        input_x = self.sigmoid(input_x)
        output = self.critic_out_layer(input_x)
        return output


class ActorCritic(Agent):
    def __init__(self,
                 action_space: pettingzoo.utils.wrappers.base,
                 agent: str,
                 gamma: float = 0.99,
                 eps_init: float = .5,
                 eps_min: float = 1e-5,
                 eps_step: float = 1e-3,
                 agent_type: str = "base",
                 name: str = "Agent",
                 load: bool = False,
                 lr_actor=1e-3,
                 lr_critics=1e-3,
                 ):
        super().__init__(
            action_space, agent, gamma, eps_init, eps_min, eps_step, agent_type,
            name, load
        )

        self.gamma = gamma

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.actor: ActorNetwork = ActorNetwork(obs_size=[6,7], action_size=7)
        self.critic: CriticNetwork = CriticNetwork(obs_size=[6,7])

        self.critic_loss_fn: nn.MSELoss = nn.MSELoss()
        self.actor_optimizer: optim.Adam = optim.Adam(
            params=self.actor.parameters(), lr=lr_actor
        )
        self.critic_optimizer: optim.Adam = optim.Adam(
            params=self.critic.parameters(), lr=lr_critics
        )

    def get_best_action(self, obs: dict) -> int:
        with torch.no_grad():
            state = torch.Tensor(obs)
            action, self.norm_dist = self.actor(state)
        return int(action)

    def update(
            self,
            obs: dict,
            action: int,
            reward: int,
            terminated: bool,
            next_obs: dict
    ):  
        observation = obs["observation"]
        observation = torch.Tensor(np.where(
            np.logical_and(
                observation[..., 0] == 0,
                observation[..., 1] == 0
            ),
            0,
            np.where(observation[..., 0] == 1, 1, -1)
        ))
        next_observation = next_obs["observation"]
        next_observation = torch.Tensor(np.where(
            np.logical_and(
                next_observation[..., 0] == 0,
                next_observation[..., 1] == 0
            ),
            0,
            np.where(next_observation[..., 0] == 1, 1, -1)
        ))
        value_state: torch.Tensor = self.critic(observation)
        value_next_state: torch.Tensor = self.critic(next_observation)
        target: float = reward + self.gamma * value_next_state.detach()
        # Calculate losses
        critic_loss: torch.Tensor = self.critic_loss_fn(value_state, target)
        actor_loss: torch.Tensor = (
            -self.norm_dist.log_prob(action).unsqueeze(0) * critic_loss.detach()
        )
        # Perform backpropagation
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def save(self, actor_checkpoint_path, critic_checkpoint_path):
        torch.save(self.actor.state_dict(), actor_checkpoint_path)
        torch.save(self.critic.state_dict(), critic_checkpoint_path)

    def load(self, actor_checkpoint_path, critic_checkpoint_path):
        self.actor.load_state_dict(
            torch.load(actor_checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.critic.load_state_dict(
            torch.load(critic_checkpoint_path, map_location=lambda storage, loc: storage)
        )
