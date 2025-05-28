#!/bin/env python3
import time
import fire
import logging
import tiledbsoma
import multiprocessing as mp

from pathlib import Path
from tiledbsoma.options import SOMATileDBContext
from tiledb import Ctx as TileDBCtx


from typing import Optional, List, Tuple
import pandas as pd
import tiledbsoma
import anndata
import scipy.sparse as sp
import concurrent.futures

logging.basicConfig(level=logging.INFO)


def group_contiguous(sorted_ids: List[int]) -> List[Tuple[int, int, List[int]]]:
    """
    Group a sorted list of integers into contiguous blocks.
    Each block is returned as a tuple (start, end, group_ids),
    where group_ids is the list of consecutive integers, and the block
    covers indices from start to end-1.
    """
    groups = []
    if not sorted_ids:
        return groups
    current = [sorted_ids[0]]
    for i in sorted_ids[1:]:
        if i == current[-1] + 1:
            current.append(i)
        else:
            groups.append((current[0], current[-1] + 1, current.copy()))
            current = [i]
    groups.append((current[0], current[-1] + 1, current.copy()))
    return groups

def read_block(ms_array, block_range: Tuple[int, int], group_ids: List[int]) -> sp.csr_matrix:
    """
    Read a contiguous slice of the measurement array and extract only the rows
    corresponding to group_ids.
    """
    start, end = block_range
    block_tensor = ms_array.read(coords=(slice(start, end),))
    # Convert to a sparse COO tensor, then to a SciPy sparse matrix, then to CSR for indexing.
    block_sparse = block_tensor.coos().concat().to_scipy().tocsr()
    # Relative indices: e.g., if block covers rows 100-150, and group_ids are [102,103,110],
    # then relative indices are [2,3,10].
    rel_idx = [jid - start for jid in group_ids]
    return block_sparse[rel_idx, :]

def get_anndata(
    exp,
    db_uri: str,
    obs_query: Optional[tiledbsoma.AxisQuery] = None,
    measurement_name: str = "RNA",
    X_name: str = "X",
    obs_columns: Optional[List[str]] = None,
    chunk_obs: bool = True,
    max_workers: int = 12
) -> anndata.AnnData:
    """
    Retrieve a subset of an experiment as an AnnData object, filtering
    the measurement (X) on-disk by only reading the rows corresponding
    to the obs query's 'soma_joinid' values.

    This function:
      1. Reads the obs metadata (optionally in chunks) and filters it via obs_query.
      2. Extracts the 'soma_joinid' values from obs and groups them into contiguous blocks.
      3. Uses a ThreadPoolExecutor to read each contiguous block (via slicing) concurrently.
      4. Vertically stacks the resulting blocks and reorders the rows to match the original obs order.
      5. Constructs an AnnData object with the filtered obs and the sparse X matrix.

    Parameters:
      db_uri (str): URI of the TileDB-SOMA experiment.
      obs_query (Optional[tiledbsoma.AxisQuery]): Query to filter observations.
      measurement_name (str): Name of the measurement (e.g. "RNA").
      X_name (str): Name of the measurement matrix (e.g. "X").
      obs_columns (Optional[List[str]]): List of columns to retrieve from obs.
      chunk_obs (bool): Whether to read obs metadata in chunks.
      max_workers (int): Maximum number of threads for parallel block reads.

    Returns:
      anndata.AnnData: The filtered AnnData object with a sparse X.
    """
    import sys
    print("Reading obs metadata...", file=sys.stderr)
    # --- Read obs metadata ---
    obs_chunks: List[pd.DataFrame] = []
    # with tiledbsoma.Experiment.open(db_uri) as exp:
    if obs_query is not None:
        try:
            reader = exp.axis_query(measurement_name, obs_query=obs_query).obs(
                column_names=obs_columns
            )
        except TypeError:
            reader = exp.axis_query(measurement_name, obs_query=obs_query).obs()
    else:
        reader = exp.obs.read(column_names=obs_columns)
    if chunk_obs:
        for chunk in reader:
            if chunk.num_rows > 0:
                obs_chunks.append(chunk.to_pandas())
        obs_df = pd.concat(obs_chunks, ignore_index=True) if obs_chunks else pd.DataFrame(columns=obs_columns)
    else:
        obs_df = reader.concat().to_pandas()

    # --- Ensure 'soma_joinid' exists and extract join IDs ---
    if "soma_joinid" not in obs_df.columns:
        raise ValueError("The obs metadata must include the 'soma_joinid' column.")
    required_ids = obs_df["soma_joinid"].tolist()
    sorted_ids = sorted(required_ids)
    groups = group_contiguous(sorted_ids)
    print(f"Found {len(groups)} contiguous blocks.", file=sys.stderr)

    config = {
        'sm.mem.total_budget': 400000000000,
        'sm.memory_budget':    200000000000,
        'sm.tile_cache_size':   50000000000,
        'sm.compute_concurrency_level': mp.cpu_count(),
        'sm.enable_prefetching': 'true',
        'sm.prefetch_data_tiles': 'true',
        'sm.io_concurrency_level': 20,
        'vfs.file.max_parallel_ops': 20,
        'vfs.min_parallel_size': 1048576,
        }
    ctx = SOMATileDBContext(tiledb_config=config)

    print("Reading measurement data by contiguous blocks in parallel...", file=sys.stderr)
    block_list = []
    # with tiledbsoma.Experiment.open(db_uri, context=ctx) as exp:
    ms_obj = exp.ms[measurement_name]
    if X_name not in ms_obj:
        available = list(ms_obj.keys())
        raise ValueError(f"Measurement '{measurement_name}' does not contain layer '{X_name}'. "
                            f"Available layers: {available}")
    ms_layer = ms_obj[X_name]
    if isinstance(ms_layer, tiledbsoma.Collection):
        keys = list(ms_layer.keys())
        ms_array = ms_layer["data"] if "data" in keys else ms_layer[keys[0]]
    else:
        ms_array = ms_layer

    # Use ThreadPoolExecutor to read blocks concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_block = {
            executor.submit(read_block, ms_array, (start, end), group_ids): (start, end, group_ids)
            for (start, end, group_ids) in groups
        }
        for future in concurrent.futures.as_completed(future_to_block):
            block_info = future_to_block[future]
            try:
                block_matrix = future.result()
                block_list.append((block_info[2], block_matrix))
                # status
                if len(block_list) % 10 == 0:
                    print(f"  Processed {len(block_list)} blocks.", file=sys.stderr)
            except Exception as e:
                raise ValueError(f"Error reading block {block_info[0]} to {block_info[1]}") from e

    if not block_list:
        raise ValueError("No measurement data was read.")

    # --- Stack blocks in ascending order of join IDs ---
    # The blocks in block_list are associated with group_ids from each block.
    # Sort blocks by the minimum join ID in each block (should already be sorted)
    print("Stacking blocks...", file=sys.stderr)
    block_list.sort(key=lambda x: x[0][0])
    blocks = [blk for (_, blk) in block_list]
    X_stacked = sp.vstack(blocks)

    # --- Reorder rows to match the original order in obs_df ---
    print("Reordering rows...", file=sys.stderr)
    id_to_pos = {jid: pos for pos, jid in enumerate(sorted_ids)}
    reorder = [id_to_pos[jid] for jid in required_ids]
    X_filtered = X_stacked[reorder, :]

    print("Reading var metadata...", file=sys.stderr)
    with tiledbsoma.Experiment.open(db_uri) as exp:
        var_reader = exp.ms[measurement_name].var.read()
        var_df = var_reader.concat().to_pandas()

    print("Creating AnnData object...", file=sys.stderr)
    adata = anndata.AnnData(X_filtered, obs=obs_df, var=var_df)
    return adata

# Example usage:
# obs_query = tiledbsoma.AxisQuery(value_filter='sample in ["smp_2743", "smp_2643"]')
# adata = get_anndata(db_uri, obs_query=obs_query, measurement_name="RNA", X_name="X")
# adata


def export(tiledb_path='/scratch/multiomics/nickyoungblut/tiledb-loader/tiledb-soma_GeneFull_Ex50pAS',
           output_path='/large_storage/ctc/datasets/scBasecamp'):
    config = {
        'sm.mem.total_budget': 400000000000,
        'sm.memory_budget':    200000000000,
        'sm.tile_cache_size':   50000000000,
        'sm.compute_concurrency_level': mp.cpu_count(),
        'sm.enable_prefetching': 'true',
        'sm.prefetch_data_tiles': 'true',
        'sm.io_concurrency_level': 20,
        'vfs.file.max_parallel_ops': 20,
        'vfs.min_parallel_size': 1048576,
        }
    ctx = SOMATileDBContext(tiledb_config=config)
    # exp = tiledbsoma.Experiment.open(tiledb_path) #, context=ctx)
    with tiledbsoma.Experiment.open(tiledb_path, context=ctx) as exp:
        df = (
                exp.obs.read(column_names=["SRX_accession"])
                .concat()
                .group_by(["SRX_accession"])
                .aggregate([
                    ([], 'count_all'),
                ])
                .sort_by([("count_all", 'ascending')])
                .to_pandas()
            )

        logging.info(f"{len(df)} files to be exported with {df['count_all'].sum()} cells in total")
        for i, row in df.iterrows():
            t0 = time.time()
            try:
                srx = row["SRX_accession"]
                output_file = Path(output_path) / f"{srx}.h5ad"
                if output_file.exists():
                    logging.info(f"Skipping {srx} as it already exists")
                    continue

                obs_query = tiledbsoma.AxisQuery(value_filter=f'SRX_accession == "{srx}"')
                # query = exp.axis_query("RNA", obs_query=obs_query)
                # logging.info(f"{i + 1} of {len(df)}: Exporting {output_file} - {[query.n_obs, query.n_vars]}...")

                # obs_query = tiledbsoma.AxisQuery(value_filter='sample in ["smp_2743", "smp_2643"]')
                adata = get_anndata(exp, tiledb_path, obs_query=obs_query, measurement_name="RNA", X_name="X")
                # adata

                # adata = query.to_anndata(X_name="data")
                adata.write(output_file)
            except Exception as e:
                logging.error(f"Error exporting {srx}: {e}")
            logging.info(f"Exported {output_file} in {time.time() - t0:.2f}s")


if __name__ == '__main__':
    fire.Fire()