import argparse


def parse_args():
    """ Parses the arguments given to the script. Will raise an exception if the arguments are invalid.
    Returns: An argparser object containing the arguments.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--gen-path", type=str, default="gs://hrgan_1/ckpnt/generator.h5", required=True)

    parser.add_argument("--images", type=str, nargs='+', default="local", required=True)

    parser.add_argument("--new-size", type=str, required=True)

    parser.add_argument("--save-to", type=str, required=True)

    return parser.parse_args()
