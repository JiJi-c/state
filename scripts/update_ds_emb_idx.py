import logging
import argparse
import anndata

from vci.data.preprocess import Preprocessor
'''
Creates a CSV file with following columns.
    #,path,species,num_cells,num_genes,names
'''


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Update gene embeddings index for a given h5ad file."
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name to be added to the dataset gene embedding mapping file",
    )
    parser.add_argument(
        "--dataset_file",
        type=str,
        required=True,
        help="H5ad file for which the gene embedding indexes has to be added to the mapping file",
    )
    parser.add_argument(
        "--emb_idx_file",
        type=str,
        default='/large_storage/ctc/public/dataset/vci/gene_embidx_mapping.torch',
        help="Path to save the output summary file",
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        default='/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
        help="Path to save the output summary file",
    )

    args = parser.parse_args()
    preprocess = Preprocessor(None, None, None, None,
                              args.embedding_file,
                              args.emb_idx_file)

    adata = anndata.read_h5ad(args.dataset_file)
    preprocess.update_dataset_emb_idx(adata, args.dataset_name)

