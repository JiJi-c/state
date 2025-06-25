import argparse as ap

from hydra import compose, initialize
from omegaconf import DictConfig

from ._cli import (
    add_arguments_emb,
    add_arguments_tx,
    run_emb_fit,
    run_emb_transform,
    run_tx_infer,
    run_tx_predict,
    run_tx_train,
)


def get_args() -> tuple[ap.Namespace, list[str]]:
    """Parse known args and return remaining args for Hydra overrides"""
    parser = ap.ArgumentParser()
    subparsers = parser.add_subparsers(required=True, dest="command")
    add_arguments_emb(subparsers.add_parser("emb"))
    add_arguments_tx(subparsers.add_parser("tx"))

    # Use parse_known_args to get both known args and remaining args
    return parser.parse_args()


def load_hydra_config(method: str, overrides: list[str] = None) -> DictConfig:
    """Load Hydra config with optional overrides"""
    if overrides is None:
        overrides = []

    # Initialize Hydra with the path to your configs directory
    # Adjust the path based on where this file is relative to configs/
    with initialize(version_base=None, config_path="configs"):
        match method:
            case "emb":
                cfg = compose(config_name="state-defaults", overrides=overrides)
            case "tx":
                cfg = compose(config_name="config", overrides=overrides)
            case _:
                raise ValueError(f"Unknown method: {method}")
    return cfg


def main():
    args = get_args()

    match args.command:
        case "emb":
            match args.subcommand:
                case "fit":
                    cfg = load_hydra_config("emb", args.hydra_overrides)
                    run_emb_fit(cfg, args)
                case "transform":
                    run_emb_transform(args)
        case "tx":
            match args.subcommand:
                case "train":
                    # Load Hydra config with overrides for sets training
                    cfg = load_hydra_config("tx", args.hydra_overrides)
                    run_tx_train(cfg)
                case "predict":
                    # For now, predict uses argparse and not hydra
                    run_tx_predict(args)
                case "infer":
                    # Run inference using argparse, similar to predict
                    run_tx_infer(args)


if __name__ == "__main__":
    main()
