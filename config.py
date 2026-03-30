import argparse
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--env-name",
        type=str,
        default="CartPole-v1",
        help="The id of gym environment",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="The learning rate for policy gradient agent.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=29,
        help="The random seed used by models and environments.",
    )
    args = parser.parse_args()
    return args


args = parse_args()
