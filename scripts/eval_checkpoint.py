import argparse
from omegaconf import OmegaConf
import torch
from vci.nn.model import LitUCEModel
from vci.nn.eval_utils import evaluate_perturbation, evaluate_de


def load_config(yaml_path):
    return OmegaConf.load(yaml_path)


import logging
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on perturbation and DE analysis.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--eval-type', type=str, choices=['perturbation', 'de', 'both'], default='both', help='Evaluation type')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger("eval_checkpoint")

    cfg = load_config(args.config)
    logger.info(f"Loading model from {args.checkpoint}")
    model = LitUCEModel.load_from_checkpoint(args.checkpoint, cfg=cfg)
    model.eval()
    model.to(args.device)

    if args.eval_type in ['perturbation', 'both']:
        logger.info("Starting Perturbation Evaluation...")
        print("\n--- Perturbation Evaluation ---")
        with tqdm(total=1, desc="Perturbation Eval", position=0) as pbar:
            evaluate_perturbation(model, cfg, device=args.device, logger=logger.info)
            pbar.update(1)
        logger.info("Finished Perturbation Evaluation.")
    if args.eval_type in ['de', 'both']:
        logger.info("Starting DE Analysis Evaluation...")
        print("\n--- DE Analysis Evaluation ---")
        with tqdm(total=1, desc="DE Analysis Eval", position=0) as pbar:
            evaluate_de(model, cfg, device=args.device, logger=logger.info)
            pbar.update(1)
        logger.info("Finished DE Analysis Evaluation.")


if __name__ == '__main__':
    main()
