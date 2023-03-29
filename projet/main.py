from projet.game import Game


def main():
    game = Game(render_mode="ansi", load_model=True, player_1_type="qlearning",
                player_2_type="qlearning")
    game.train(epoch=100_000, verbose=1, save=True)
