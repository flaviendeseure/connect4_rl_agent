import argparse

from pettingzoo.classic import connect_four_v3
from pettingzoo.utils import OrderEnforcingWrapper

from projet.agent import ActorCritic, Random, Human,MCTS
from projet.agent.base_agent import Agent
from projet.game import Game


def get_player(player_type: str, space, name: str, load: bool = False) -> Agent | Human:
    if player_type == "actor_critic":
        return ActorCritic(space, name, load=load, name=name)
    elif player_type == "random":
        return Random(space, name, name=name)
    elif player_type == "human":
        return Human()
    elif player_type == "mcts":
        return MCTS(space, name)
    else:
        raise Exception("Invalid player")


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument("--train", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--eval", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--play", action=argparse.BooleanOptionalAction,
                        default=False)
    parser.add_argument("--epoch", type=int, default=100_000)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--load", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--player_0", type=str, default="actor_critic")
    parser.add_argument("--player_1", type=str, default="actor_critic")
    args = parser.parse_args()

    if (args.play and args.train) or (args.play and args.eval) or (
            args.train and args.eval):
        raise Exception("You can't do multiple actions at the same time")

    if not args.play and not args.train and not args.eval:
        raise Exception("You need to specify an action")

    if args.play:
        env: OrderEnforcingWrapper = connect_four_v3.env(render_mode="human")
        space = env.action_space
        player_0: Agent | Human = get_player(
            args.player_0, space, "player_0", load=args.load
        )
        player_1: Agent | Human = get_player(
            args.player_1, space, "player_1", load=args.load
        )
        if not isinstance(player_0, Human) and not isinstance(player_1, Human):
            raise Exception("You can't play with two agents")
        game = Game(env, player_0, player_1)
        game.play()
        return

    env: OrderEnforcingWrapper = connect_four_v3.env(render_mode="ansi")
    space = env.action_space
    player_0 = get_player(args.player_0, space, "player_0", load=args.load)
    player_1 = get_player(args.player_1, space, "player_1", load=args.load)
    game = Game(env, player_0, player_1)

    if args.train:
        game.train(epoch=args.epoch, verbose=args.verbose, save=args.save)
    elif args.eval:
        game.eval(nb_eval=args.epoch, verbose=args.verbose)


if __name__ == "__main__":
    main()
