import pickle
import time
from pathlib import Path

from pettingzoo.classic import connect_four_v3
from tqdm import tqdm

from projet.player import QLearner, Human, Random, Sarsa


class Game:
    def __init__(self, render_mode: str = "ansi", load_model: bool = False,
                 model_name: str = "qlearning", player_1_type: str = "qlearning",
                 player_2_type: str = "qlearning") -> None:
        self.env = connect_four_v3.env(render_mode=render_mode)
        self.player_0, self.player_1 = self._create_players(player_1_type,
                                                            player_2_type)
        self.models_dir = Path("projet/models") / model_name
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.player_0_path: Path = self.models_dir / "player_0.pkl"
        self.player_1_path: Path = self.models_dir / "player_1.pkl"
        self.path: Path = self.models_dir / "model.pkl"
        if load_model:
            self.load()

    def _create_players(self, player_1_type: str, player_2_type: str) -> list:
        players: list = []
        self.env.reset()
        for agent in self.env.agent_iter():
            if agent == "player_0":
                if player_1_type == "qlearning":
                    players.append(QLearner(self.env.action_space, agent))
                elif player_1_type == "random":
                    players.append(Random(self.env.action_space, agent))
                elif player_1_type == "human":
                    players.append(Human())
                elif player_1_type == "sarsa":
                    players.append(Sarsa(self.env.action_space, agent))
            else:
                if player_2_type == "qlearning":
                    players.append(QLearner(self.env.action_space, agent))
                elif player_2_type == "random":
                    players.append(Random(self.env.action_space, agent))
                elif player_2_type == "human":
                    players.append(Human())
                elif player_2_type == "sarsa":
                    players.append(Sarsa(self.env.action_space, agent))
            self.env.step(0)
            if agent == "player_1":
                break
        self.env.reset()
        return players

    def train(self, epoch: int, verbose: int = 0, save: bool = True) -> None:
        if isinstance(self.player_0, Human) or isinstance(self.player_1, Human):
            raise Exception("You can't train with a human player")

        nb_wins_agent_1: int = 0
        nb_wins_agent_2: int = 0
        nb_draws: int = 0

        for i in tqdm(range(epoch)):
            self.env.reset()
            for agent in self.env.agent_iter():
                last_observation, reward, termination, truncation, info = self.env.last()
                if termination:
                    if reward == 1 and agent == "player_0":
                        nb_wins_agent_1 += 1
                    elif reward == 1 and agent == "player_1":
                        nb_wins_agent_2 += 1
                    elif reward == 0 and agent == "player_0":
                        nb_draws += 1
                    if i % 1000 == 0 and verbose and i != 0 and agent == "player_0":
                        print(
                            f"Agent 1 wins: {nb_wins_agent_1}, Agent 2 wins: {nb_wins_agent_2}, Draws: {nb_draws}, Ratio: {nb_wins_agent_1 / (nb_wins_agent_2 + nb_wins_agent_1) : .2f}"
                        )

                    # if verbose:
                    #     print(f"Termination ({agent}), Reward: {reward}, info: {info}")
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

                    observation, reward, termination, truncation, info = self.env.last()
                    if agent == "player_0":
                        self.player_0.update(last_observation, action, reward,
                                             termination, observation)
                    else:
                        self.player_1.update(last_observation, action, reward,
                                             termination, observation)

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
                last_observation, reward, termination, truncation, info = self.env.last()
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
            last_observation, reward, termination, truncation, info = self.env.last()
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

                observation, reward, termination, truncation, info = self.env.last()
                if agent == "player_0" and not isinstance(self.player_0, Human):
                    self.player_0.update(last_observation, action, reward,
                                         termination, observation)
                elif agent == "player_1" and not isinstance(self.player_1, Human):
                    self.player_1.update(last_observation, action, reward,
                                         termination, observation)
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
        q_values = {**self.player_0.q_values, **self.player_1.q_values}
        pickle.dump(q_values, open(self.path, "wb"))

    def load(self):
        if not self.path.exists():
            return

        q_values = pickle.load(open(self.path, "rb"))
        if not isinstance(self.player_0, Human):
            self.player_0.q_values = q_values
        if not isinstance(self.player_1, Human):
            self.player_1.q_values = q_values
