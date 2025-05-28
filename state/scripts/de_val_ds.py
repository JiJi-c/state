#!/bin/env python3

'''
Filters out cells with top 5000 expressed genes
'''

import scanpy as sc
import numpy as np

n_top_genes = 500
input_file = '/large_storage/ctc/datasets/vci/validation/rpe1.h5ad'
output_file = f'/large_storage/ctc/datasets/vci/validation/rpe1_top{n_top_genes}_variable.h5ad'


adata = sc.read_h5ad(input_file)

sc.pp.highly_variable_genes(adata, n_top_genes=5000, flavor='seurat_v3')
adata = adata[:, adata.var['highly_variable']]

adata.write(output_file)