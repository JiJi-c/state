import os
import anndata

from omegaconf import OmegaConf

from vci.inference import Inference
from vci.data import create_dataloader


def test_simple_infernce():
    config_file = f'/home/aadduri/vci_pretrain/outputs/lowlr_bce/conf/training.yaml'
    chkp_file = f'/large_storage/ctc/userspace/aadduri/vci/checkpoint/lowlr_bce/exp_lowlr_bce_layers_8_dmodel_512_samples_2048_max_lr_0.00024_op_dim_512-epoch=0-step=277000.ckpt'

    input_file = '/large_storage/ctc/datasets/vci/validation/replogle_perturbation.h5ad'
    dataset_name = os.path.basename(input_file).split('.')[0]

    cfg = OmegaConf.load(config_file)

    inferer = Inference(config_file)
    inferer.load_model(chkp_file)
    adata = anndata.read_h5ad(input_file)
    dataset_name = os.path.basename(input_file).split('.')[0]
    dataloader = create_dataloader(cfg,
                                   adata=adata,
                                   adata_name=dataset_name)

    for embeddings in inferer.encode(dataloader):
        print(embeddings.shape)

