from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

import numpy as np
import pettingzoo


class Agent(ABC):
    def __init__(
        self,
        action_space: pettingzoo.utils.wrappers.base,
        agent: str,
        gamma: float = 0.99,
        eps_init: float = 0.5,
        eps_min: float = 1e-5,
        eps_step: float = 1e-3,
        agent_type: str = "actor_critic",
        name: str = "Agent",
    ):

        self.action_space: pettingzoo.utils.wrappers.base = action_space
        self.agent: str = agent
        self.gamma: float = gamma
        self.eps: float = eps_init
        self.eps_min: float = eps_min
        self.eps_step: float = eps_step

        self.models_dir: Path = Path("projet/models") / agent_type
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.player_path: Path = self.models_dir / name

        self.reset()

    def eps_greedy(self, obs: dict, eps: Union[float, None] = None) -> int:
        eps = eps or self.eps
        if np.random.random() < eps:
            return self.action_space(self.agent).sample(mask=obs["action_mask"])
        else:
            return self.get_best_action(obs)

    def get_action(self, obs: dict, eps: Union[float, None] = None) -> int:
        return self.eps_greedy(obs, eps)

    def epsilon_decay(self) -> None:
        self.eps = max(self.eps - self.eps_step, self.eps_min)

    @abstractmethod
    def get_best_action(self, obs: dict) -> int:
        raise NotImplementedError

    @abstractmethod
    def update(
        self, obs: dict, action: int, reward: float, terminated: bool, next_obs: dict
    ):
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError
