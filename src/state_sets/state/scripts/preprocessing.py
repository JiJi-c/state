import logging
import argparse

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
        description="Create dataset list CSV file"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        # required=True,
        default ='/home/yhr/scRecount/data/recount/mouse',
        #default='/common_datasets/external/references/cellxgene',
        help="Directory containing all H5AD files for training",
    )
    parser.add_argument(
        "--destination",
        type=str,
        # required=True,
        default='/large_storage/ctc/ML/data/cell/recount/processed',
        help="Directory to store the processed files",
    )
    parser.add_argument(
        "--summary_file",
        type=str,
        # required=True,
        default='/scratch/ctc/ML/vci/h5ad_recount.csv',
        help="Path to save the output summary file",
    )
    parser.add_argument(
        "--emb_idx_file",
        type=str,
        # required=True,
        default='/scratch/ctc/ML/uce/model_files/gene_embidx_mapping.torch',
        help="Path to save the output summary file",
    )
    parser.add_argument(
        "--embedding_file",
        type=str,
        # required=True,
        default='/large_storage/ctc/ML/data/cell/misc/Homo_sapiens.GRCh38.gene_symbol_to_embedding_ESM2.pt',
        help="Path to save the output summary file",
    )
    parser.add_argument(
        "--species",
        type=str,
        # required=True,
        default='mouse',
        help="Path to save the output summary file",
    )

    args = parser.parse_args()
    preprocess = Preprocessor(args.species,
                              args.data_path,
                              args.destination,
                              args.summary_file,
                              args.embedding_file,
                              args.emb_idx_file)
    preprocess.process()
