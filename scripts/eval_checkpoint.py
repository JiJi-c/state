import argparse
import yaml
import torch
from vci.nn.model import LitUCEModel
from vci.nn.eval_utils import evaluate_perturbation, evaluate_de


def load_config(yaml_path):
    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)
    # Optionally, convert to an object with attribute access
    # For now, use as dict
    from munch import Munch
    return Munch.fromDict(cfg)


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on perturbation and DE analysis.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--eval-type', type=str, choices=['perturbation', 'de', 'both'], default='both', help='Evaluation type')
    args = parser.parse_args()

    cfg = load_config(args.config)
    model = LitUCEModel.load_from_checkpoint(args.checkpoint, cfg=cfg)
    model.eval()
    model.to(args.device)

    if args.eval_type in ['perturbation', 'both']:
        print("\n--- Perturbation Evaluation ---")
        evaluate_perturbation(model, cfg, device=args.device)
    if args.eval_type in ['de', 'both']:
        print("\n--- DE Analysis Evaluation ---")
        evaluate_de(model, cfg, device=args.device)

if __name__ == '__main__':
    main()
