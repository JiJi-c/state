import argparse as ap


def add_arguments_sets(parser: ap.ArgumentParser):
    """"""
    subparsers = parser.add_subparsers(required=True, dest="subcommand")
    subparsers.add_parser("train")
    subparsers.add_parser("predict")
