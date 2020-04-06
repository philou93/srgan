import csv

from tester.view_loss_history.argsparser import parse_args
from tester.view_loss_history.display import *


def main(args):
    generator_history = []
    discriminator_history = []

    with open(args.history_path, 'r', newline='\n') as csv_file:
        history_reader = csv.reader(csv_file, delimiter=',')
        for i, history in enumerate(history_reader):
            if i % args.step_per_epoch == 0:
                # On regroupe les loss par epoch
                generator_history.append([])
                discriminator_history.append([])
            generator_history[-1].append(history[0])
            discriminator_history[-1].append(history[1])

    if args.sgl:
        show_single_model_loss(generator_history, "Generator")
    if args.sdl:
        show_single_model_loss(discriminator_history, "Discriminator")


if __name__ == "__main__":
    args = parse_args()

    main(args)
