from projet.game import Game


def main():
    game = Game(render_mode="ansi", load_model=True, player_1_type="sarsa",
                player_2_type="random")
    game.train(epoch=5001, verbose=1, save=True)
