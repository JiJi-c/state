import argparse as ap

from ._predict import add_arguments_predict
from ._train import add_arguments_train


def add_arguments_sets(parser: ap.ArgumentParser):
    """"""
    subparsers = parser.add_subparsers(required=True, dest="subcommand")
    add_arguments_train(subparsers.add_parser("train"))
    add_arguments_predict(subparsers.add_parser("predict"))
