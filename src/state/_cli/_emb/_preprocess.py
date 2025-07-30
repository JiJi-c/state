import argparse as ap
import os


def add_arguments_preprocess(parser: ap.ArgumentParser):
    """Add arguments for embedding preprocessing CLI."""
    parser.add_argument(
        "--profile-name", required=True,
        help="Name for the new profile (used for embeddings and dataset)"
    )
    parser.add_argument(
        "--train-csv", required=True,
        help="Path to training CSV file (species,path,names columns)"
    )
    parser.add_argument(
        "--val-csv", required=True,
        help="Path to validation CSV file (species,path,names columns)"
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to output generated files"
    )
    parser.add_argument(
        "--config-file", default=None,
        help="Config file to update (default: src/state/configs/state-defaults.yaml)"
    )
    parser.add_argument(
        "--all-embeddings", default=None,
        help="Path to existing all_embeddings.pt file (if not provided, creates one-hot embeddings)"
    )


def run_emb_preprocess(args):
    """
    Preprocess datasets and embeddings to create a new profile.
    """
    import logging
    import sys
    import pandas as pd
    import torch
    from omegaconf import OmegaConf
    from tqdm import tqdm

    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    # Default config file
    if args.config_file is None:
        args.config_file = "src/state/configs/state-defaults.yaml"

    # Prepare output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load or create embeddings
    if args.all_embeddings:
        if not os.path.exists(args.all_embeddings):
            log.error(f"All embeddings file not found: {args.all_embeddings}")
            sys.exit(1)
        log.info(f"Loading existing embeddings from {args.all_embeddings}")
        all_embeddings = torch.load(args.all_embeddings)
        if not isinstance(all_embeddings, dict):
            log.error("All embeddings file must be a dict mapping gene names to tensors")
            sys.exit(1)
        # normalize keys
        all_embeddings = {str(k).upper(): v for k, v in all_embeddings.items()}
        emb_size = next(iter(all_embeddings.values())).shape[0]
        gene_to_idx = {g: i for i, g in enumerate(all_embeddings.keys())}
        use_onehot = False
    else:
        all_embeddings = None
        gene_to_idx = {}
        emb_size = None
        use_onehot = True

    # Load CSVs
    log.info("Loading training and validation CSV files...")
    try:
        train_df = pd.read_csv(args.train_csv)
        val_df = pd.read_csv(args.val_csv)
    except Exception as e:
        log.error(f"Error loading CSV files: {e}")
        sys.exit(1)

    # Validate CSV columns
    for name, df in [("train", train_df), ("val", val_df)]:
        missing = set(["species", "path", "names"]) - set(df.columns)
        if missing:
            log.error(f"{name} CSV missing required columns: {missing}")
            sys.exit(1)

    log.info(f"Processing {len(train_df)} training and {len(val_df)} validation datasets...")

    # Collect dataset info and all genes
    dataset_info = {}
    all_genes = set()

    def process_df(df, label):
        nonlocal all_genes, dataset_info
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {label}" ):
            name = row["names"]
            path = row["path"]
            if not os.path.exists(path):
                log.error(f"Dataset file not found: {path}")
                sys.exit(1)
            # detect gene field per file
            gene_field = detect_gene_name_strategy(path)
            log.info(f"{label} {name}: using gene field '{gene_field}'")
            num_cells, num_genes, genes = extract_dataset_info(path, gene_field)
            dataset_info[name] = {"num_cells": num_cells, "num_genes": num_genes, "genes": genes}
            all_genes.update(genes)

    process_df(train_df, "training")
    process_df(val_df, "validation")

    log.info(f"Found {len(all_genes)} unique genes across datasets")

    # Create embeddings and masks
    valid_genes_masks = {}
    if use_onehot:
        log.info("Creating one-hot embeddings...")
        all_embeddings = create_onehot_embeddings(all_genes)
        emb_size = len(all_genes)
        gene_to_idx = {g: i for i, g in enumerate(sorted(all_genes))}
        for name, info in dataset_info.items():
            valid_genes_masks[name] = torch.ones(len(info["genes"]), dtype=torch.bool)
    else:
        missing = []
        for name, info in dataset_info.items():
            genes = info["genes"]
            mask = torch.tensor([g in all_embeddings for g in genes], dtype=torch.bool)
            valid_genes_masks[name] = mask
            missing.extend([g for g, ok in zip(genes, mask) if not ok])
        if missing:
            log.error(f"Missing genes in embeddings: {sorted(set(missing))[:20]} (+{len(set(missing))-20})")
            sys.exit(1)

    # Build ds_emb_mapping
    ds_emb_mapping = {
        name: torch.tensor([gene_to_idx[g] for g in info["genes"]], dtype=torch.long)
        for name, info in dataset_info.items()
    }

    # Save outputs
    emb_file = os.path.join(args.output_dir, f"all_embeddings_{args.profile_name}.pt")
    torch.save(all_embeddings, emb_file)
    map_file = os.path.join(args.output_dir, f"ds_emb_mapping_{args.profile_name}.torch")
    torch.save(ds_emb_mapping, map_file)
    masks_file = os.path.join(args.output_dir, f"valid_genes_masks_{args.profile_name}.torch")
    torch.save(valid_genes_masks, masks_file)
    log.info("Saved embeddings, mapping, and masks")

    # Write updated CSVs
    train_out = create_updated_csv(train_df, dataset_info, args.output_dir, f"train_{args.profile_name}.csv")
    val_out = create_updated_csv(val_df, dataset_info, args.output_dir, f"val_{args.profile_name}.csv")

    # Update config
    total_ds = len(train_df) + len(val_df)
    update_config_file(
        args.config_file,
        args.profile_name,
        emb_file,
        map_file,
        masks_file,
        train_out,
        val_out,
        emb_size,
        len(all_embeddings),
        total_ds,
    )

    log.info("Preprocessing completed. Run: uv run state emb fit --conf %s embeddings.current=%s dataset.current=%s",
             args.config_file, args.profile_name, args.profile_name)


def detect_gene_name_strategy(dataset_path):
    """Detect which var field under /var/ holds gene names."""
    import h5py as h5
    with h5.File(dataset_path, 'r') as f:
        if 'var' not in f:
            raise ValueError(f"No var/ in {dataset_path}")
        for fld in ['_index', 'gene_name', 'gene_symbols', 'feature_name', 'gene_id', 'symbol']:
            if fld in f['var']:
                return fld
    raise ValueError(f"No gene field found in var/ of {dataset_path}")


def extract_dataset_info(dataset_path, gene_field):
    """Extract num_cells, num_genes, and uppercase gene list."""
    import h5py as h5
    with h5.File(dataset_path, 'r') as f:
        X = f['X']
        attrs = dict(X.attrs)
        if 'encoding-type' in attrs:
            enc = attrs['encoding-type']
            if enc in ('csr_matrix', 'csc_matrix'):
                num_cells, num_genes = attrs['shape']
            else:
                num_cells, num_genes = X.shape
        else:
            if hasattr(X, 'shape') and len(X.shape) == 2:
                num_cells, num_genes = X.shape
            else:
                num_cells = len(X['indptr'])-1
                num_genes = int(X['indices'][:].max())+1
        # read genes
        grp = f['var'][gene_field]
        raw = grp['categories'][:] if 'categories' in grp else grp[:]
        genes = [item.decode('utf-8').upper() if isinstance(item, (bytes,bytearray)) else str(item).upper() for item in raw]
        if len(genes) != num_genes:
            raise ValueError(f"Gene count mismatch {len(genes)} vs {num_genes}")
    return num_cells, num_genes, genes


def create_onehot_embeddings(all_genes):
    """Make one-hot embeddings for each gene."""
    import torch
    genes_sorted = sorted(all_genes)
    emb = {}
    for i, g in enumerate(genes_sorted):
        vec = torch.zeros(len(genes_sorted))
        vec[i] = 1.0
        emb[g] = vec
    return emb


def create_updated_csv(df, info_map, out_dir, filename):
    """Add num_cells, num_genes, groupid_for_de and save."""
    import pandas as pd
    out = df.copy()
    out['num_cells'] = out['names'].map(lambda n: info_map[n]['num_cells'])
    out['num_genes'] = out['names'].map(lambda n: info_map[n]['num_genes'])
    out['groupid_for_de'] = 'leiden'
    path = os.path.join(out_dir, filename)
    out.to_csv(path, index=False)
    return path


def update_config_file(config_path, profile_name,
                       embeddings_file, mapping_file, masks_file,
                       train_csv, val_csv,
                       embedding_size, num_embeddings, num_datasets):
    """Insert new profile into config YAML."""
    from omegaconf import OmegaConf
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
    else:
        cfg = OmegaConf.create({})
    if 'embeddings' not in cfg:
        cfg.embeddings = {}
    if 'dataset' not in cfg:
        cfg.dataset = {}
    cfg.embeddings[profile_name] = {
        'all_embeddings': embeddings_file,
        'ds_emb_mapping': mapping_file,
        'valid_genes_masks': masks_file,
        'size': embedding_size,
        'num': num_embeddings,
    }
    cfg.dataset[profile_name] = {
        'ds_type': 'h5ad',
        'train': train_csv,
        'val': val_csv,
        'filter': False,
        'num_datasets': num_datasets,
    }
    with open(config_path, 'w') as f:
        OmegaConf.save(cfg, f)
