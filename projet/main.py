from pettingzoo.classic import connect_four_v3
from pettingzoo.utils import OrderEnforcingWrapper

from projet.agent import QLearner
from projet.game import Game


def main():
    env: OrderEnforcingWrapper = connect_four_v3.env(render_mode="ansi")
    space = env.action_space
    player_0 = QLearner(space, "player_0")
    player_1 = QLearner(space, "player_1")

    game = Game(env, player_0, player_1)
    game.train(epoch=1_000_001, verbose=1, save=True)
