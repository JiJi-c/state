#!/usr/bin/env python

import argparse
import os
import sys
import pickle
import re
import yaml
import logging
import anndata
import scanpy as sc
import numpy as np
import pandas as pd
import torch
import wandb

from lightning.pytorch import Trainer

# Import the relevant modules from your repository
from models.decoders import UCELogProbDecoder
from data.mapping_strategies import (
    CentroidMappingStrategy,
    ClusteringMappingStrategy,
    BatchMappingStrategy,
    RandomMappingStrategy,
    NearestNeighborMappingStrategy,
    PseudoBulkMappingStrategy,
)
from data.data_modules import MultiDatasetPerturbationDataModule
from validation.metrics import compute_metrics
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    """
    CLI for evaluation.
    """
    parser = argparse.ArgumentParser(description="Evaluate a trained PerturbationModel.")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output_dir containing the config.yaml file that was saved during training.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="last.ckpt",
        help="Checkpoint filename. Default is 'last.ckpt'. Relative to the output directory.",
    )
    parser.add_argument(
        "--map_type",
        type=str,
        default=None,
        choices=[
            "centroid",
            "clustering",
            "batch",
            "random",
            "nearest",
            "pseudo_nearest",
            "pseudobulk",
        ],
        help="Override the mapping strategy at inference time (optional).",
    )
    parser.add_argument(
        "--test_time_finetune",
        type=int,
        default=0,
        help="Number of epochs to fine-tune on control cells from test set before evaluation (default: 0, no fine-tuning)",
    )
    return parser.parse_args()


def load_config(cfg_path: str) -> dict:
    """
    Load config from the YAML file that was dumped during training.
    """
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Could not find config file: {cfg_path}")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def get_latest_step_checkpoint(directory):
    # Get all checkpoint files
    files = os.listdir(directory)
    
    # Extract step numbers using regex, excluding files with 'val_loss'
    step_numbers = []
    for f in files:
        if f.startswith('step=') and 'val_loss' not in f:
            match = re.search(r'step=(\d+)(?:-v\d+)?\.ckpt', f)
            if match:
                step_numbers.append(int(match.group(1)))
    
    if not step_numbers:
        raise ValueError("No checkpoint files found")
        
    max_step = max(step_numbers)
    checkpoint_path = os.path.join(directory, f"step={max_step}.ckpt")
    return checkpoint_path


def process_test_chunks(model, test_loader, data_module, cfg, chunk_size=500000):
    """
    Processes the test set in chunks of approximately 'chunk_size' cells.
    For each chunk, the predictions and metadata are accumulated, converted into AnnData objects,
    and metrics are computed using your existing compute_metrics function.
    
    Returns:
        A list of tuples, each containing (chunk_metrics, chunk_cell_counts)
        where chunk_metrics is the dict returned by compute_metrics,
        and chunk_cell_counts is a dict mapping cell types to number of cells in that chunk.
    """
    model.eval()
    device = next(model.parameters()).device
    # Accumulators for the current chunk
    chunk_acc = {
        "preds": [],
        "X": [],
        "X_gene": [],
        "pert_name": [],
        "cell_type": [],
        "gem_group": [],
        "basal": []
    }
    total_cells = 0
    chunk_metrics_list = []  # List to hold (metrics, counts) for each chunk
    pbar = tqdm(
        total=len(test_loader),
        desc="Testing",
        unit="batch",
        leave=True,
        position=0,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )

    for batch in test_loader:
        # Transfer batch to device
        batch = data_module.transfer_batch_to_device(batch, device, dataloader_idx=0)
        with torch.no_grad():
            output = model(batch)
        # Here we assume your model's forward returns predictions.
        # (If you use predict_step which returns a dict, adjust accordingly.)
        batch_preds = output.cpu().numpy() if output is not None else None
        batch_X = batch["X"].cpu().numpy() if ("X" in batch and batch["X"] is not None) else None
        batch_X_gene = batch["X_gene"].cpu().numpy() if ("X_gene" in batch and batch["X_gene"] is not None) else None

        # For metadata, assume they are either lists or tensors
        batch_pert = batch["pert_name"] if "pert_name" in batch else None
        batch_celltype = batch["cell_type"] if "cell_type" in batch else None
        batch_gem = batch["gem_group"] if "gem_group" in batch else None
        batch_basal = batch["basal"].cpu().numpy() if ("basal" in batch and batch["basal"] is not None) else None

        # Determine number of cells in this batch (using preds if available)
        if batch_preds is not None:
            num_cells = batch_preds.shape[0]
        elif batch_X is not None:
            num_cells = batch_X.shape[0]
        else:
            num_cells = 0

        total_cells += num_cells

        # Accumulate data for this chunk
        if batch_preds is not None:
            chunk_acc["preds"].append(batch_preds)
        if batch_X is not None:
            chunk_acc["X"].append(batch_X)
        if batch_X_gene is not None:
            chunk_acc["X_gene"].append(batch_X_gene)
        if batch_pert is not None:
            if isinstance(batch_pert, torch.Tensor):
                chunk_acc["pert_name"].extend(batch_pert.cpu().numpy().tolist())
            else:
                chunk_acc["pert_name"].extend(batch_pert)
        if batch_celltype is not None:
            if isinstance(batch_celltype, torch.Tensor):
                chunk_acc["cell_type"].extend(batch_celltype.cpu().numpy().tolist())
            else:
                chunk_acc["cell_type"].extend(batch_celltype)
        if batch_gem is not None:
            if isinstance(batch_gem, torch.Tensor):
                chunk_acc["gem_group"].extend(batch_gem.cpu().numpy().tolist())
            else:
                chunk_acc["gem_group"].extend(batch_gem)
        if batch_basal is not None:
            chunk_acc["basal"].append(batch_basal)

        # When accumulated cells exceed the chunk size, process the chunk.
        if total_cells >= chunk_size:
            # Concatenate arrays for numerical data
            concat_preds = np.concatenate(chunk_acc["preds"], axis=0) if chunk_acc["preds"] else None
            concat_X = np.concatenate(chunk_acc["X"], axis=0) if chunk_acc["X"] else None
            concat_X_gene = np.concatenate(chunk_acc["X_gene"], axis=0) if chunk_acc["X_gene"] else None
            # Build metadata DataFrame
            metadata = pd.DataFrame({
                "pert_name": chunk_acc["pert_name"],
                "cell_type": chunk_acc["cell_type"],
                "gem_group": chunk_acc["gem_group"]
            })
            # Build AnnData objects
            adata_pred = anndata.AnnData(X=concat_preds, obs=metadata.copy())
            adata_real = anndata.AnnData(X=concat_X, obs=metadata.copy())
            adata_real_exp = None
            if concat_X_gene is not None:
                adata_real_exp = anndata.AnnData(X=concat_X_gene, obs=metadata.copy())
                # Optionally, set var.index for gene names if available.
            
            # Compute metrics for this chunk
            chunk_metrics = compute_metrics(
                adata_pred=adata_pred,
                adata_real=adata_real,
                adata_real_exp=adata_real_exp,
                include_dist_metrics=True,
                control_pert=data_module.get_control_pert(),
                pert_col="pert_name",
                celltype_col="cell_type",
                model_loc=None,
                DE_metric_flag=False,
                class_score_flag=True,
                embed_key=data_module.embed_key,
                transform=data_module.transform,
                output_space=cfg["data"]["kwargs"]["output_space"],
                decoder=model.decoder,
                shared_perts=data_module.get_shared_perturbations(),
            )
            # Also compute cell counts per cell type for this chunk
            chunk_counts = metadata.groupby("cell_type").size().to_dict()
            chunk_metrics_list.append((chunk_metrics, chunk_counts))
            # Reset the accumulators and cell count for the next chunk
            chunk_acc = {key: [] for key in chunk_acc}
            total_cells = 0

        pbar.update(1)

    # Process any leftover data in the accumulators.
    if total_cells > 0:
        concat_preds = np.concatenate(chunk_acc["preds"], axis=0) if chunk_acc["preds"] else None
        concat_X = np.concatenate(chunk_acc["X"], axis=0) if chunk_acc["X"] else None
        # concat_X_gene = np.concatenate(chunk_acc["X_gene"], axis=0) if chunk_acc["X_gene"] else None
        concat_X_gene = None
        metadata = pd.DataFrame({
            "pert_name": chunk_acc["pert_name"],
            "cell_type": chunk_acc["cell_type"],
            "gem_group": chunk_acc["gem_group"]
        })
        adata_pred = anndata.AnnData(X=concat_preds, obs=metadata.copy())
        adata_real = anndata.AnnData(X=concat_X, obs=metadata.copy())
        adata_real_exp = None
        if concat_X_gene is not None:
            adata_real_exp = anndata.AnnData(X=concat_X_gene, obs=metadata.copy())
        chunk_metrics = compute_metrics(
            adata_pred=adata_pred,
            adata_real=adata_real,
            adata_real_exp=adata_real_exp,
            include_dist_metrics=True,
            control_pert=data_module.get_control_pert(),
            pert_col="pert_name",
            celltype_col="cell_type",
            model_loc=None,
            DE_metric_flag=False,
            class_score_flag=True,
            embed_key=data_module.embed_key,
            transform=data_module.transform,
            output_space=cfg["data"]["kwargs"]["output_space"],
            decoder=model.decoder,
            shared_perts=data_module.get_shared_perturbations(),
        )
        chunk_counts = metadata.groupby("cell_type").size().to_dict()
        chunk_metrics_list.append((chunk_metrics, chunk_counts))
    pbar.close()
    
    return chunk_metrics_list


def aggregate_chunk_metrics(chunk_metrics_list):
    """
    Aggregates metrics across chunks using a weighted average.
    For each cell type and metric, each chunk's metric is weighted by the number of cells in that cell type.
    
    Returns:
        A dictionary mapping cell types to a dictionary of averaged metric values.
    """
    aggregated = {}  # celltype -> {metric_name: weighted_sum, '_count': total_count}
    for chunk_metrics, chunk_counts in chunk_metrics_list:
        for celltype, df in chunk_metrics.items():
            count = chunk_counts.get(celltype, 0)
            if count == 0:
                continue
            if celltype not in aggregated:
                aggregated[celltype] = {"_count": 0}
            aggregated[celltype]["_count"] += count
            for metric in df.columns:
                # Here we take the mean of the metric in the chunk (df[metric].mean())
                val = df[metric].mean()
                if metric not in aggregated[celltype]:
                    aggregated[celltype][metric] = 0
                aggregated[celltype][metric] += val * count
    final_metrics = {}
    for celltype, metrics in aggregated.items():
        total_count = metrics.pop('_count')
        final_metrics[celltype] = {m: (v / total_count) for m, v in metrics.items()}
    return final_metrics


def main():
    args = parse_args()

    # 1. Load the config
    config_path = os.path.join(args.output_dir, "config.yaml")
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # 2. Find run output directory & check for data_module.pkl
    run_output_dir = os.path.join(cfg["output_dir"], cfg["name"])
    data_module_path = os.path.join(run_output_dir, "data_module.pkl")
    if not os.path.exists(data_module_path):
        raise FileNotFoundError(
            f"Could not find serialized data module at {data_module_path}.\n"
            "Did you remember to pickle it in the training script?"
        )

    # 3. Load the data module
    with open(data_module_path, "rb") as f:
        data_module: MultiDatasetPerturbationDataModule = pickle.load(f)
    logger.info("Loaded data module from %s", data_module_path)

    # If user overrides the mapping strategy
    if args.map_type is not None:
        mapping_cls = {
            "centroid": CentroidMappingStrategy,
            "clustering": ClusteringMappingStrategy,
            "batch": BatchMappingStrategy,
            "random": RandomMappingStrategy,
            "nearest": NearestNeighborMappingStrategy,
            "pseudobulk": PseudoBulkMappingStrategy,
        }[args.map_type]
        strategy_kwargs = {
            "random_state": cfg["training"]["train_seed"],
            "n_basal_samples": cfg["data"]["kwargs"]["n_basal_samples"],
            "k_neighbors": cfg["data"]["kwargs"].get("k_neighbors", 10),
            "neighborhood_fraction": cfg["data"]["kwargs"].get("neighborhood_fraction", 0.0),
        }
        data_module.set_inference_mapping_strategy(mapping_cls, **strategy_kwargs)
        logger.info("Inference mapping strategy set to %s", args.map_type)

    # 4. Load the trained model
    checkpoint_dir = os.path.join(run_output_dir, "checkpoints")
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Could not find checkpoint at {checkpoint_path}.\nSpecify a correct checkpoint filename with --checkpoint."
        )
    logger.info("Loading model from %s", checkpoint_path)

    model_class_name = cfg["model"]["name"]
    model_kwargs = cfg["model"]["kwargs"]

    if model_class_name.lower() == "embedsum":
        from models.embed_sum import EmbedSumPerturbationModel
        ModelClass = EmbedSumPerturbationModel
    elif model_class_name.lower() == "neuralot":
        from models.neural_ot import NeuralOTPerturbationModel
        ModelClass = NeuralOTPerturbationModel
    elif model_class_name.lower() == "simplesum":
        from models.simple_sum import SimpleSumPerturbationModel
        ModelClass = SimpleSumPerturbationModel
    elif model_class_name.lower() == "globalsimplesum":
        from models.global_simple_sum import GlobalSimpleSumPerturbationModel
        ModelClass = GlobalSimpleSumPerturbationModel
    else:
        raise ValueError(f"Unknown model class: {model_class_name}")

    var_dims = data_module.get_var_dims()
    model_init_kwargs = {
        "input_dim": var_dims["input_dim"],
        "hidden_dim": model_kwargs["hidden_dim"],
        "output_dim": var_dims["output_dim"],
        "pert_dim": var_dims["pert_dim"],
        **model_kwargs,
    }

    model = ModelClass.load_from_checkpoint(checkpoint_path, **model_init_kwargs)
    model.eval()
    logger.info("Model loaded successfully.")

    # 5. Prepare the test dataloader.
    data_module.setup(stage="test")
    data_module.cell_sentence_len = cfg["model"]["kwargs"]["transformer_backbone_kwargs"]["n_positions"]
    test_loader = data_module.test_dataloader()
    if test_loader is None:
        logger.warning("No test dataloader found. Exiting.")
        sys.exit(0)

    if args.test_time_finetune > 0:
        from train.test_time_finetuner import TestTimeFineTuner
        logger.info(f"Running test-time fine-tuning for {args.test_time_finetune} epochs...")
        finetuner = TestTimeFineTuner(
            model=model,
            test_loader=test_loader,
            control_pert=data_module.get_control_pert(),
            num_epochs=args.test_time_finetune,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        finetuner.finetune()

    # 6. Process test data in chunks and compute metrics per chunk.
    logger.info("Processing test set in chunks...")
    # chunk_metrics_list = process_test_chunks(model, test_loader, data_module, cfg, chunk_size=500000)
    chunk_metrics_list = process_test_chunks(model, test_loader, data_module, cfg, chunk_size=50000)
    aggregated_metrics = aggregate_chunk_metrics(chunk_metrics_list)

    # 7. Summarize results.
    summary = []
    for celltype, metric_dict in aggregated_metrics.items():
        for metric_name, value in metric_dict.items():
            summary.append({
                "celltype": celltype,
                "metric_name": metric_name,
                "metric_val": value
            })
    summary_df = pd.DataFrame(summary)
    logger.info("Summary of aggregated metrics:\n" + str(summary_df))

    # 8. Save metrics to CSV.
    eval_basedir = os.path.join(
        run_output_dir,
        "eval",
        f"map_{args.map_type or 'train_map'}",
        args.checkpoint.replace(".ckpt", ""),
    )
    os.makedirs(eval_basedir, exist_ok=True)
    metrics_csv_path = os.path.join(eval_basedir, "metrics.csv")
    summary_df.to_csv(metrics_csv_path, index=False)
    logger.info(f"Metrics saved to {metrics_csv_path}")

    # Optionally store results in wandb.
    wandb_info_path = os.path.join(run_output_dir, "wandb_path.txt")
    if os.path.exists(wandb_info_path):
        with open(wandb_info_path, "r") as f:
            wandb_run_path = f.read().strip()
        try:
            api = wandb.Api()
            eval_run = api.run(wandb_run_path)
            logger.info(f"Logging summary metrics to wandb run {wandb_run_path}")
            for _, row in summary_df.iterrows():
                metric_key = f"test/{row['metric_name']}_{row['celltype']}"
                eval_run.summary[metric_key] = float(row["metric_val"])
                metric_key = f"test/{row['metric_name']}"
                eval_run.summary[metric_key] = float(row["metric_val"])
            eval_run.summary.update()
            logger.info("Metrics updated in wandb run.")
        except wandb.CommError:
            logger.warning("Failed to connect to wandb run. No wandb logging done.")
    else:
        logger.info("No wandb_path.txt found; skipping wandb update.")


if __name__ == "__main__":
    main()