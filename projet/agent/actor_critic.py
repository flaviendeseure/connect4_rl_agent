from pathlib import Path

import numpy as np
import pettingzoo
import torch
import torch.distributions as dist
import torch.optim as optim
from torch import nn

from projet.agent.base_agent import Agent

HIDDEN_SIZE = 256  # 128


class ActorNetwork(nn.Module):
    def __init__(self, action_size: int) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.flatten = nn.Flatten(start_dim=0)
        self.fc1_layer: nn.Linear = nn.Linear(384, HIDDEN_SIZE)
        self.fc2_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.actor_out_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, action_size)

    def forward(
            self, input_x: torch.Tensor
    ) -> torch.Tensor:
        input_x = nn.functional.relu(self.conv_1(input_x))
        input_x = nn.functional.relu(self.conv_2(input_x))
        input_x = self.flatten(input_x)
        input_x = self.fc1_layer(input_x)
        input_x = self.fc2_layer(input_x)
        input_x = nn.functional.softmax(self.actor_out_layer(input_x), dim=0)
        return input_x


class CriticNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv_1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=0)
        self.conv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=0)
        self.flatten = nn.Flatten(start_dim=0)
        self.fc1_layer: nn.Linear = nn.Linear(384, HIDDEN_SIZE)
        self.fc2_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.critic_out_layer: nn.Linear = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, input_x: torch.Tensor) -> torch.Tensor:
        input_x = nn.functional.relu(self.conv_1(input_x))
        input_x = nn.functional.relu(self.conv_2(input_x))
        input_x = self.flatten(input_x)
        input_x = nn.functional.relu(self.fc1_layer(input_x))
        input_x = nn.functional.relu(self.fc2_layer(input_x))
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
                 agent_type: str = "actor_critic",
                 name: str = "Agent",
                 load: bool = False,
                 lr_actor=1e-3,
                 lr_critics=1e-3,
                 ):
        super().__init__(action_space, agent, gamma, eps_init, eps_min, eps_step,
                         agent_type, name)
        self.actor_path = self.models_dir / Path(f"{name}_actor.pth")
        self.critic_path = self.models_dir / Path(f"{name}_critic.pth")

        self.gamma = gamma

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.actor: ActorNetwork = ActorNetwork(action_size=7)
        self.critic: CriticNetwork = CriticNetwork()

        #self.critic_loss_fn: nn.MSELoss = nn.MSELoss()
        self.actor_optimizer: optim.Adam = optim.Adam(
            params=self.actor.parameters(), lr=lr_actor
        )
        self.critic_optimizer: optim.Adam = optim.Adam(
            params=self.critic.parameters(), lr=lr_critics
        )

        if load:
            self.load()

    def get_best_action(self, obs: dict) -> int:
        observation = obs["observation"]
        observation = torch.Tensor(np.where(
            np.logical_and(
                observation[..., 0] == 0,
                observation[..., 1] == 0
            ),
            0,
            np.where(observation[..., 0] == 1, 1, -1)
        )).unsqueeze(0)
        with torch.no_grad():
            state = torch.Tensor(observation)
            action = self.actor(state)
            action = dist.Categorical(action)
            action = action.sample()
            return action.item()

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
        )).unsqueeze(0)
        next_observation = next_obs["observation"]
        next_observation = torch.Tensor(np.where(
            np.logical_and(
                next_observation[..., 0] == 0,
                next_observation[..., 1] == 0
            ),
            0,
            np.where(next_observation[..., 0] == 1, 1, -1)
        )).unsqueeze(0)

        probs = self.actor(observation)
        dist = torch.distributions.Categorical(probs=probs)
        action = dist.sample()

        advantage = reward + (1.0 - terminated) * self.gamma * self.critic(next_observation) - self.critic(observation)

        critic_loss = advantage.pow(2).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -dist.log_prob(action) * advantage.detach()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        

    def save(self):
        torch.save(self.actor.state_dict(), self.actor_path)
        torch.save(self.critic.state_dict(), self.critic_path)

    def load(self):
        if not self.actor_path.exists() or not self.critic_path.exists():
            print("No model to load")
        else:
            self.actor.load_state_dict(
                torch.load(self.actor_path, map_location=lambda storage, loc: storage)
            )
            self.critic.load_state_dict(
                torch.load(self.critic_path, map_location=lambda storage, loc: storage)
            )

    def reset(self):
        pass