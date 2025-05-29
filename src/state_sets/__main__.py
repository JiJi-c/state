import argparse as ap

from ._cli import add_arguments_sets, add_arguments_state


def get_args() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    add_arguments_state(subparsers.add_parser("state"))
    add_arguments_sets(subparsers.add_parser("sets"))
    return parser.parse_args()


def main():
    args = get_args()
    print(args)
    pass


if __name__ == "__main__":
    main()
