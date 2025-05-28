#!/usr/bin/env python
import argparse
import scanpy as sc
from inference_module import InferenceModule


def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using a trained model.")
    parser.add_argument(
        "--model_folder",
        type=str,
        required=True,
        help="Path to the model folder (containing data_module.pkl and checkpoint).",
    )
    parser.add_argument("--input_anndata", type=str, required=True, help="Path to the input anndata (.h5ad) file.")
    parser.add_argument(
        "--output_anndata",
        type=str,
        required=True,
        help="Path where the output anndata with predictions will be saved.",
    )
    parser.add_argument(
        "--pert_key", type=str, default="gene", help="Column key name in anndata with perturbation labels."
    )
    parser.add_argument(
        "--celltype_key", type=str, default="cell_type", help="Column key name in anndata with cell type labels."
    )
    parser.add_argument(
        "--cell_set_len",
        type=int,
        default=32,
        help="Length of the cell set to use for perturbation. Larger is typically better.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("Loading model from:", args.model_folder)
    # Instantiate the inference module with the model class that was used during training.
    inference_module = InferenceModule(model_folder=args.model_folder, cell_set_len=args.cell_set_len)

    print("Loading input anndata from:", args.input_anndata)
    adata = sc.read_h5ad(args.input_anndata)

    print("Running inference...")
    adata_out = inference_module.perturb(
        adata,
        pert_key=args.pert_key,
        celltype_key=args.celltype_key,
    )
    adata_out.write_h5ad(args.output_anndata)
    print("Inference complete. Output saved to:", args.output_anndata)


if __name__ == "__main__":
    main()
