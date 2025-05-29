import argparse as ap

from ._embed import add_arguments_embed
from ._train import add_arguments_train


def add_arguments_state(parser: ap.ArgumentParser):
    """"""
    subparsers = parser.add_subparsers(required=True, dest="subcommand")

    add_arguments_train(subparsers.add_parser("train"))
    add_arguments_embed(subparsers.add_parser("embed"))
