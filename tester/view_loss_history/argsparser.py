import argparse


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--history-path", type=str, default="save/loss_history.csv", required=False)
    parser.add_argument("--step-per-epoch", type=int, default=100, required=False)

    # Affiche le graphique de la loss du generateur
    parser.add_argument("--sgl", type=bool, default=True, required=False)
    # Affiche le graphique de la loss du discriminateur
    parser.add_argument("--sdl", type=bool, default=True, required=False)

    parse_args = parser.parse_args()

    return parser.parse_args()
