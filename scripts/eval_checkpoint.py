import argparse
from omegaconf import OmegaConf
import torch
from vci.nn.model import LitUCEModel
from vci.nn.eval_utils import evaluate_intrinsic, evaluate_de


def load_config(yaml_path):
    return OmegaConf.load(yaml_path)


import logging
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on intrinsic and DE analysis.")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda or cpu)')
    parser.add_argument('--eval-type', type=str, choices=['intrinsic', 'de', 'both'], default='both', help='Evaluation type')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for evaluation')
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )
    logger = logging.getLogger("eval_checkpoint")

    cfg = load_config(args.config)
    cfg.model.batch_size = args.batch_size
    logger.info(f"Loading model from {args.checkpoint}")
    from vci.inference import Inference
    inference = Inference(cfg)
    inference.load_model(args.checkpoint)
    model = inference.model
    model.to(args.device)
    # inference handles pe_embedding and eval mode

    pred_adata = None
    if args.eval_type in ['de', 'both']:
        logger.info("Starting DE Analysis Evaluation...")
        print("\n--- DE Analysis Evaluation ---")
        with tqdm(total=1, desc="DE Analysis Eval", position=0) as pbar:
            pred_adata = evaluate_de(model, cfg, device=args.device, logger=logger.info)
            pbar.update(1)
        logger.info("Finished DE Analysis Evaluation.")

    if args.eval_type in ['intrinsic', 'both']:
        logger.info("Starting Intrinsic Evaluation...")
        print("\n--- Intrinsic Evaluation ---")
        with tqdm(total=1, desc="Intrinsic Eval", position=0) as pbar:
            # we evaluate on the same adata for both
            evaluate_intrinsic(model, cfg, device=args.device, logger=logger.info, adata=pred_adata)
            pbar.update(1)
        logger.info("Finished Intrinsic Evaluation.")
    


if __name__ == '__main__':
    main()
