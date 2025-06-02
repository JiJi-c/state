import argparse as ap


def add_arguments_embed(parser: ap.ArgumentParser):
    """Add arguments for state embedding CLI."""
    parser.add_argument("--checkpoint", required=True, help="Path to the model checkpoint file")
    parser.add_argument("--config", required=True, help="Path to the model training config")
    parser.add_argument("--input", required=True, help="Path to input anndata file (h5ad)")
    parser.add_argument("--output", required=True, help="Path to output embedded anndata file (h5ad)")
    parser.add_argument("--dataset-name", default="perturbation", help="Dataset name to be used in dataloader creation")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--filter", action="store_true", help="Filter gene set to our esm embeddings only.")
    parser.add_argument("--embed-key", help="Name of key to store embeddings")


def run_state_embed(args: ap.ArgumentParser):
    """
    Compute embeddings for an input anndata file using a pre-trained VCI model checkpoint.
    """
    import os
    import logging

    from omegaconf import OmegaConf

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    from ...state.inference import Inference

    # Load configuration
    logger.info(f"Loading config from {args.config}")
    conf = OmegaConf.load(args.config)

    # Create inference object
    logger.info("Creating inference object")
    inferer = Inference(conf)

    # Load model from checkpoint
    logger.info(f"Loading model from checkpoint: {args.checkpoint}")
    inferer.load_model(args.checkpoint)

    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

    # Generate embeddings
    logger.info(f"Computing embeddings for {args.input}")
    logger.info(f"Output will be saved to {args.output}")

    inferer.encode_adata(
        input_adata_path=args.input,
        output_adata_path=args.output,
        emb_key=args.embed_key,
        dataset_name=args.dataset_name,
    )

    logger.info("Embedding computation completed successfully!")
