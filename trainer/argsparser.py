import argparse


def parse_args():
    """ Parses the arguments given to the script. Will raise an exception if the arguments are invalid.
    Returns: An argparser object containing the arguments.

    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--extension-file", nargs='+', default=["jpg", "png"], required=False)

    parser.add_argument("--ckpnt-gen", type=str, default="gs://srgan_2/ckpnt/generator.h5", required=False)
    parser.add_argument("--ckpnt-discr", type=str, default="gs://srgan_2/ckpnt/discriminator.h5", required=False)
    parser.add_argument("--history-path", type=str, default="gs://srgan_2/ckpnt/history.csv", required=False)

    parser.add_argument("--weights-discr-path", type=str, required=False)
    parser.add_argument("--weights-gen-path", type=str, required=False)

    parser.add_argument("--epoch", type=int, default=30, required=False)
    parser.add_argument("--step", type=int, default=100, required=False)
    parser.add_argument("--batch_size", type=int, default=4, required=False)
    parser.add_argument("--lr-factor", type=int, default=4, required=False)

    # Simule le fait qu'on est sur google clud pour les paths
    parser.add_argument("--location", type=str, default="gcloud", choices=['local', 'gcloud'], required=False)

    parser.add_argument("--job-dir", required=False)

    return parser.parse_args()
