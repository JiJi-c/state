import argparse as ap

from ._cli import (
    add_arguments_sets,
    add_arguments_state,
    run_sets_predict,
    run_sets_train,
    run_state_embed,
    run_state_train,
)


def get_args() -> ap.ArgumentParser:
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    add_arguments_state(subparsers.add_parser("state"))
    add_arguments_sets(subparsers.add_parser("sets"))
    return parser.parse_args()


def main():
    args = get_args()
    match args.command:
        case "state":
            match args.subcommand:
                case "train":
                    run_state_train(args)
                case "embed":
                    run_state_embed(args)
        case "sets":
            match args.subcommand:
                case "train":
                    run_sets_train(args)
                case "predict":
                    run_sets_predict(args)


if __name__ == "__main__":
    main()
