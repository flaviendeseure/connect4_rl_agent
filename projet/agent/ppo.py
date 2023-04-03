import pettingzoo
import torch
from torch import nn

from projet.agent import RolloutBuffer, ActorCritic
from projet.agent.base_agent import Agent


class PPO(Agent):
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
                 obs_dim=None,
                 lr_actor_critics=None,
                 ):
        super().__init__(
            action_space, agent, gamma, eps_init, eps_min, eps_step, agent_type,
            name, load
        )

        if obs_dim is None:
            obs_dim = [7, 2]
        if lr_actor_critics is None:
            lr_actor_critics = [1e-4, 1e-3]

        self.gamma = gamma
        self.eps_clip = eps_init
        self.lr_actor, self.lr_critic = lr_actor_critics

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(obs_dim, len(action_space)).to(self.device)
        self.optimizer = torch.optim.Adam(
            [
                {"params": self.policy.actor.parameters(), "lr": self.lr_actor},
                {"params": self.policy.critic.parameters(), "lr": self.lr_critic},
            ]
        )

        self.policy_old = ActorCritic(obs_dim, len(action_space)).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def get_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_clipped, action, action_logprob, state_val = self.policy_old.act(
                state)

        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action_clipped

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(
                reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)
        ):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(
            self.device)
        old_actions = torch.squeeze(
            torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(
            torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = (
            torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(
                self.device)
        )

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for 5 epochs
        for _ in range(5):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states,
                                                                        old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip,
                                1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = (
                    -torch.min(surr1, surr2)
                    + 0.5 * self.MseLoss(state_values, rewards)
                    - 0.01 * dist_entropy
            )

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
        self.policy.load_state_dict(
            torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        )
