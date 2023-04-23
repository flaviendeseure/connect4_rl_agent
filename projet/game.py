import time
from typing import Union

from pettingzoo.classic import connect_four_v3
from pettingzoo.utils import OrderEnforcingWrapper
from tqdm import tqdm

from projet.agent import Human
from projet.agent.base_agent import Agent


class Game:
    def __init__(
        self,
        env: OrderEnforcingWrapper,
        player_0: Union[Agent, Human],
        player_1: Union[Agent, Human],
    ) -> None:
        self.env = env
        self.player_0: Agent = player_0
        self.player_1: Agent = player_1

    def train(self, epoch: int, verbose: int = 0, save: bool = True) -> None:
        if isinstance(self.player_0, Human) or isinstance(self.player_1, Human):
            raise Exception("You can't train with a human player")

        nb_wins_agent_1: int = 0
        nb_wins_agent_2: int = 0
        nb_draws: int = 0

        for i in tqdm(range(epoch)):
            self.env.reset()
            for agent in self.env.agent_iter():
                (
                    last_observation,
                    reward,
                    termination,
                    truncation,
                    info,
                ) = self.env.last()
                if termination:
                    if reward == 1 and agent == "player_0":
                        nb_wins_agent_1 += 1
                    elif reward == 1 and agent == "player_1":
                        nb_wins_agent_2 += 1
                    elif reward == 0 and agent == "player_0":
                        nb_draws += 1
                    if i % 500 == 0 and verbose and i != 0 and agent == "player_0":
                        print(
                            f"Agent 1 wins: {nb_wins_agent_1}, Agent 2 wins: {nb_wins_agent_2}, Draws: {nb_draws}, Ratio: {100 * (nb_wins_agent_1 / (nb_wins_agent_2 + nb_wins_agent_1)) : .2f}"
                        )

                    self.env.step(None)
                elif truncation:
                    if verbose:
                        print("Truncated")
                else:  # we update the actor and critic networks weights every steps
                    if agent == "player_0":
                        action = self.player_0.get_action(last_observation)
                    else:
                        action = self.player_1.get_action(last_observation)
                    self.env.step(action)

                    observation, reward, termination, truncation, info = self.env.last()
                    if agent == "player_0":
                        self.player_0.update(
                            last_observation, action, reward, termination, observation
                        )
                    else:
                        self.player_1.update(
                            last_observation, action, reward, termination, observation
                        )

            if save and i % 1000 == 0:
                self.save()
        self.save()

    def eval(self, nb_eval: int, verbose: int = 0):
        if isinstance(self.player_0, Human) or isinstance(self.player_1, Human):
            raise Exception("You can't eval with a human player")

        nb_wins_agent_1: int = 0
        nb_wins_agent_2: int = 0
        nb_draws: int = 0

        for i in tqdm(range(nb_eval)):
            self.env.reset()
            for agent in self.env.agent_iter():
                (
                    last_observation,
                    reward,
                    termination,
                    truncation,
                    info,
                ) = self.env.last()
                if termination:
                    if reward == 1 and agent == "player_0":
                        nb_wins_agent_1 += 1
                    elif reward == 1 and agent == "player_1":
                        nb_wins_agent_2 += 1
                    elif reward == 0 and agent == "player_0":
                        nb_draws += 1
                    if i % 100 == 0 and verbose and i != 0 and agent == "player_0":
                        print(
                            f"Agent 1 wins: {nb_wins_agent_1}, Agent 2 wins: {nb_wins_agent_2}, Draws: {nb_draws}"
                        )
                    self.env.step(None)
                elif truncation:
                    if verbose:
                        print("Truncated")
                else:
                    if agent == "player_0":
                        action = self.player_0.get_action(last_observation)
                    else:
                        action = self.player_1.get_action(last_observation)
                    self.env.step(action)

        print(
            f"Agent 1 wins: {nb_wins_agent_1}, Agent 2 wins: {nb_wins_agent_2}, Draws: {nb_draws}"
        )

    def play(self):
        self.env.reset()
        for agent in self.env.agent_iter():
            last_observation, reward, termination, truncation, _ = self.env.last()
            if termination:
                self.env.step(None)
                if reward == 1:
                    print(f"Game finished, {agent} won")
                elif reward == -1:
                    print(f"Game finished, {agent} lost")
                else:
                    print("Game finished, draw")
            elif truncation:
                print("Truncated")
            else:
                if agent == "player_0":
                    action = self.player_0.get_action(last_observation)
                else:
                    action = self.player_1.get_action(last_observation)
                self.env.step(action)

                observation, reward, termination, truncation, _ = self.env.last()
                if agent == "player_0" and not isinstance(self.player_0, Human):
                    self.player_0.update(
                        last_observation, action, reward, termination, observation
                    )
                elif agent == "player_1" and not isinstance(self.player_1, Human):
                    self.player_1.update(
                        last_observation, action, reward, termination, observation
                    )
                self.env.render()

    def watch(self):
        env = self.env
        self.env = connect_four_v3.env(render_mode="human")
        self.env.reset()
        for agent in self.env.agent_iter():
            last_observation, reward, termination, truncation, info = self.env.last()
            if termination:
                print(f"Termination ({agent}), Reward: {reward}, info: {info}")
                self.env.step(None)
            elif truncation:
                print("Truncated")
            else:
                if agent == "player_0":
                    action = self.player_0.get_action(last_observation)
                else:
                    action = self.player_1.get_action(last_observation)
                self.env.step(action)
                self.env.render()
                time.sleep(2)
        self.env = env

    def save(self):
        self.player_0.save()
        self.player_1.save()
