from pettingzoo.classic import connect_four_v3
from pettingzoo.utils import OrderEnforcingWrapper

from projet.agent import QLearner, ActorCritic, Random, Human
from projet.game import Game


def main():
    env: OrderEnforcingWrapper = connect_four_v3.env(render_mode="human")
    space = env.action_space
    player_0 = ActorCritic(space, "player_0", load=True)
    player_1 = Human()

    game = Game(env, player_0, player_1)
    game.play()
    #game.eval(100)
    """game.train(epoch=500, verbose=1, save=True)

    print("Training finished")
    player_0 = ActorCritic(space, "player_0", load=True)
    player_1 = Human()
    print("------------------")
    game = Game(env, player_0, player_1)
    game.play()"""
