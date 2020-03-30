import argparse


def parse_args():
    """ Parses the arguments given to the script. Will raise an exception if the arguments are invalid.
    Returns: An argparser object containing the arguments.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--gen-path", type=str, default="gs://hrgan_1/ckpnt/generator.h5", required=True)
    parser.add_argument("--discr-path", type=str, default="gs://hrgan_1/ckpnt/discriminator.h5", required=False)

    parser.add_argument("--images", type=str, nargs='+', default="local", required=True)

    parser.add_argument("--factor", type=int, default=4, required=False)

    parse_args = parser.parse_args()

    return parser.parse_args()
