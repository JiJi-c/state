import os
import h5py
import argparse
import random
from typing import Dict, Set, List

def load_perturbations_by_celltype(h5_files: List[str], cell_type_key: str, pert_key: str) -> Dict[str, Set[str]]:
    """
    Load perturbations organized by cell type from specified h5 files.
    
    Args:
        h5_files: List of paths to .h5 files
        cell_type_key: Key for cell type info in /obs/{cell_type_key}
        pert_key: Key for perturbation data in h5 files
        
    Returns:
        Dictionary mapping cell type to set of perturbations
    """
    ct_to_perts = {}
    
    for f in h5_files:
        if not os.path.exists(f):
            print(f"Warning: File {f} does not exist, skipping...")
            continue
            
        try:
            with h5py.File(f, 'r') as h5_file:
                print(f"Processing file: {f}")
                
                # Get cell types for all observations in this file
                cell_type_codes = h5_file[f'/obs/{cell_type_key}/codes'][:]
                cell_type_categories = h5_file[f'/obs/{cell_type_key}/categories'][:]
                cell_types = cell_type_categories[cell_type_codes]
                
                # Get perturbations for all observations in this file
                pert_codes = h5_file[f'/obs/{pert_key}/codes'][:]
                pert_categories = h5_file[f'/obs/{pert_key}/categories'][:]
                perts = pert_categories[pert_codes]
                
                # Convert bytes to strings if needed
                if len(cell_types) > 0 and isinstance(cell_types[0], bytes):
                    cell_types = [ct.decode('utf-8') for ct in cell_types]
                
                if len(perts) > 0 and isinstance(perts[0], bytes):
                    perts = [p.decode('utf-8') for p in perts]
                
                # Ensure arrays have same length
                if len(cell_types) != len(perts):
                    print(f"Warning: Length mismatch in {f}: {len(cell_types)} cell types, {len(perts)} perturbations")
                    continue
                
                # Get unique cell types in this file
                unique_cell_types = set(cell_types)
                print(f"  Found cell types: {unique_cell_types}")
                
                # For each cell type, collect perturbations from masked subset
                for cell_type in unique_cell_types:
                    # Create mask for this cell type
                    mask = [ct == cell_type for ct in cell_types]
                    
                    # Get perturbations for this cell type
                    cell_type_perts = [perts[i] for i, is_match in enumerate(mask) if is_match]
                    
                    # Add to global dictionary
                    if cell_type not in ct_to_perts:
                        ct_to_perts[cell_type] = set()
                    ct_to_perts[cell_type].update(cell_type_perts)
                
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")
            continue
    
    return ct_to_perts

def split_perturbations(
    target_perts: Set[str],
    val_frac: float,
    test_frac: float,
    random_seed: int = 42
) -> tuple[list[str], list[str], list[str]]:
    """
    Split target cell type perturbations into train/val/test sets.
    
    Args:
        target_perts: All perturbations available in target cell type
        val_frac: Fraction of perturbations to use for validation
        test_frac: Fraction of perturbations to use for test
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_perturbations, val_perturbations, test_perturbations)
    """
    all_perts = list(target_perts)
    
    if len(all_perts) == 0:
        print("Warning: No perturbations found for target cell type!")
        return [], [], []
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    random.shuffle(all_perts)
    
    # Calculate number of perturbations for each set
    total_perts = len(all_perts)
    num_val = int(val_frac * total_perts)
    num_test = int(test_frac * total_perts)
    
    # Ensure fractions don't exceed 1.0
    if val_frac + test_frac > 1.0:
        print(f"Warning: val_frac ({val_frac}) + test_frac ({test_frac}) > 1.0")
        print("Adjusting to use proportional split...")
        total_frac = val_frac + test_frac
        num_val = int((val_frac / total_frac) * total_perts)
        num_test = int((test_frac / total_frac) * total_perts)
    
    # Split perturbations
    val_perts = all_perts[:num_val]
    test_perts = all_perts[num_val:num_val + num_test]
    train_perts = all_perts[num_val + num_test:]
    
    return train_perts, val_perts, test_perts

def main():
    parser = argparse.ArgumentParser(description='Split target cell type perturbations into train/val/test sets')
    parser.add_argument('--h5_files', type=str, nargs='+', required=True,
                      help='List of paths to .h5 files to process')
    parser.add_argument('--cell_type', type=str, required=True,
                      help='Target cell type for test set')
    parser.add_argument('--cell_type_key', type=str, default='cell_type',
                      help='Key for cell type info in h5 files')
    parser.add_argument('--pert_key', type=str, default='perturbations',
                      help='Key for perturbation data in h5 files')
    parser.add_argument('--val_frac', type=float, default=0.1,
                      help='Fraction of target cell type perturbations for validation set')
    parser.add_argument('--test_frac', type=float, default=0.1,
                      help='Fraction of target cell type perturbations for test set')
    parser.add_argument('--random_seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Load perturbations by cell type
    print(f"Loading perturbations from {len(args.h5_files)} files...")
    print(f"Files to process: {args.h5_files}")
    print(f"Using cell_type_key: {args.cell_type_key}, pert_key: {args.pert_key}")
    ct_to_perts = load_perturbations_by_celltype(args.h5_files, args.cell_type_key, args.pert_key)
    
    if not ct_to_perts:
        print("Error: No perturbation data found!")
        return
    
    print(f"\nFound {len(ct_to_perts)} cell types:")
    for ct, perts in ct_to_perts.items():
        print(f"  {ct}: {len(perts)} perturbations")
    
    # Check if target cell type exists
    if args.cell_type not in ct_to_perts:
        print(f"Error: Cell type '{args.cell_type}' not found in data!")
        print(f"Available cell types: {list(ct_to_perts.keys())}")
        return
    
    # Get perturbations for target cell type
    target_perts = ct_to_perts[args.cell_type]
    print(f"\nTarget cell type '{args.cell_type}' has {len(target_perts)} perturbations")
    other_perts = set()
    for ct in ct_to_perts:
        if ct == args.cell_type:
            continue
        other_perts |= ct_to_perts[ct]

    old_target_len = len(target_perts)
    target_perts &= other_perts
    filtered_perts = old_target_len - len(target_perts) # for perturbations not found in training data, we just train on them
    print(f"\nTarget cell type '{args.cell_type}' has {len(target_perts)} perturbations also seen in other cts")

    
    # Split target cell type perturbations into train/val/test
    train_perts, val_perts, test_perts = split_perturbations(
        target_perts,
        args.val_frac,
        args.test_frac,
        args.random_seed
    )
    
    # Print results
    print(f"\nSplit summary for '{args.cell_type}':")
    print(f"  Training: {filtered_perts + len(train_perts)} perturbations")
    print(f"  Validation: {len(val_perts)} perturbations") 
    print(f"  Test: {len(test_perts)} perturbations")
    
    # Also print as Python lists for easy copy-paste
    print(f"\n# Validation set:")
    print(f"val_perturbations = {val_perts}")
    
    print(f"\n# Test set:")
    print(f"test_perturbations = {test_perts}")

if __name__ == "__main__":
    main()