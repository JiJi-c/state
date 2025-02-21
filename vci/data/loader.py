import os
import h5py
import logging
import torch
import functools
import torch.utils.data as data
import numpy as np

from typing import Dict

from torch.utils.data import DataLoader
import vci.utils as utils
import pandas as pd

log = logging.getLogger(__file__)


def create_dataloader(cfg,
                      batch_size=32,
                      workers=1,
                      data_dir=None,
                      datasets=None,
                      shape_dict=None,
                      adata=None,
                      adata_name=None,
                      shuffle=True,
                      sentence_collator=None):
        '''
        Expected to be used for inference
        Either datasets and shape_dict or adata and adata_name should be provided
        '''
        if datasets is None and adata is None:
            raise ValueError('Either datasets and shape_dict or adata and adata_name should be provided')

        if data_dir:
            cfg.dataset.data_dir = data_dir

        dataset = H5adDatasetSentences(cfg,
                                       datasets=datasets,
                                       shape_dict=shape_dict,
                                       adata=adata,
                                       adata_name=adata_name)
        if sentence_collator is None:
            sentence_collator = VCIDatasetSentenceCollator(cfg)
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                shuffle=shuffle,
                                collate_fn=sentence_collator,
                                num_workers=workers,
                                persistent_workers=True)
        return dataloader


class H5adDatasetSentences(data.Dataset):
    def __init__(self,
                 cfg,
                 test=False,
                 datasets=None,
                 shape_dict=None,
                 adata=None,
                 adata_name=None) -> None:
        super(H5adDatasetSentences, self).__init__()

        self.adata = None
        self.adata_name = adata_name
        self.test = test
        if adata is not None:
            self.adata = adata
            self.datasets = [adata_name]
            self.shapes_dict = {self.datasets[0]: adata.shape}
        elif datasets is None:
            ds_path = cfg.dataset.train
            if test:
                ds_path = cfg.dataset.val
            _, self.datasets, self.shapes_dict, self.dataset_path_map, self.dataset_group_map= utils.get_shapes_dict(ds_path)
        else:
            assert shape_dict is not None
            assert len(datasets) == len(shape_dict)
            self.datasets = datasets
            self.shapes_dict = shape_dict

        self.datasets = sorted(self.datasets)
        self.cfg = cfg

        self.num_cells = {}
        self.num_genes = {}

        self.total_num_cells = 0
        for name in self.datasets:
            num_cells, num_genes = self.shapes_dict[name]
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes

            self.total_num_cells += num_cells

        self.datasets_to_num = {
            k: v for k, v in zip(self.datasets, range(len(self.datasets)))
        }

    def _compute_index(self, idx):
        for dataset in self.datasets:
            if idx < self.num_cells[dataset]:
                return dataset, idx
            else:
                idx -= self.num_cells[dataset]
        raise IndexError

    @functools.lru_cache
    def dataset_file(self, dataset):
        datafile = self.dataset_path_map[dataset]
        return h5py.File(datafile, "r")

    def _get_DE_scores(self, h5f, idx, de_group):

        cluster_id = str(h5f[f'/obs/{de_group}/codes'][idx])
        if de_group != 'leiden':
            cluster_id = h5f[f'/obs/{de_group}/categories'][int(cluster_id)].decode('utf-8')

        gene_indices = torch.tensor(h5f['/uns/ranked_genes/gene_indices'][cluster_id][:])
        gene_scores = torch.tensor(h5f['/uns/ranked_genes/gene_scores'][cluster_id][:])
        gene_scores = torch.nn.functional.softmax(gene_scores)
        # gene_scores = torch.nn.functional.softplus(gene_scores)
        return gene_indices, gene_scores

    def __getitem__(self, idx):
        if isinstance(idx, int):
            if self.adata is not None:
                # block is only used during validation
                counts = torch.tensor(self.adata.X[idx].todense())
                dataset = self.adata_name
                dataset_num = 0

                de_group = 'leiden'
                group_id = self.adata.obs[de_group][idx]
                ranked_genes = self.adata.uns['ranked_genes']['gene_indices'].columns

                if group_id not in ranked_genes:
                    raise KeyError(f"Gene '{group_id}' missing in ranked_genes.")

                gene_indices = torch.tensor(self.adata.uns['ranked_genes']['gene_indices'][group_id].to_numpy())
                gene_scores = torch.tensor(self.adata.uns['ranked_genes']['gene_scores'][group_id].to_numpy())
                gene_scores = torch.nn.functional.softmax(gene_scores)
                return counts, idx, dataset, dataset_num, gene_indices, gene_scores

            dataset, ds_idx = self._compute_index(idx)
            h5f = self.dataset_file(dataset)
            attrs = dict(h5f['X'].attrs)
            try:
                if attrs['encoding-type'] == 'csr_matrix':
                    indptrs = h5f["/X/indptr"]
                    start_ptr = indptrs[ds_idx]
                    end_ptr = indptrs[ds_idx + 1]
                    sub_data = torch.tensor(
                        h5f["/X/data"][start_ptr:end_ptr],
                        dtype=torch.float)
                    sub_indices = torch.tensor(
                        h5f["/X/indices"][start_ptr:end_ptr],
                        dtype=torch.int32)

                    counts = torch.sparse_csr_tensor(
                        [0,],
                        sub_indices,
                        sub_data,
                        (1, self.num_genes[dataset]),
                    )
                    counts = counts.to_dense()
                else:
                    log.info('debugging', ds_idx, 'end')
                    log.info(ds_idx)
                    counts = torch.tensor(h5f["X"][ds_idx]).unsqueeze(0)
                gene_indices, gene_scores = self._get_DE_scores(h5f, ds_idx, self.dataset_group_map[dataset])
                if gene_indices is None or gene_scores is None:
                    return None
            except IndexError as iex:
                log.exception(f"Error in dataset {dataset} at index {ds_idx}")
                raise iex

            dataset_num = self.datasets_to_num[dataset]
            return counts, idx, dataset, dataset_num, gene_indices, gene_scores
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes

class FilteredGenesCounts(H5adDatasetSentences):
    def __init__(self, cfg, test=False, datasets=None, shape_dict=None, adata=None, adata_name=None) -> None:
        super(FilteredGenesCounts, self).__init__(cfg, test, datasets, shape_dict, adata, adata_name)
        self.valid_gene_index = {}
        if cfg.embeddings.esm2.embedding_file is not None:
            esm_data = torch.load(cfg.embeddings.esm2.embedding_file)
            valid_genes_list = list(esm_data.keys())
            for name in self.datasets:
                if not utils.is_valid_uuid(name): # had to add this in for now as cellxgene h5ad fles don't have gene_name object but tahoe does
                    a = self.dataset_file(name)
                    gene_names = np.array([g.decode('utf-8') for g in a["/var/gene_name"][:]])  # Decode byte strings
                    valid_mask = np.isin(gene_names, valid_genes_list)
                    self.valid_gene_index[name] = valid_mask
                    num_valid_genes = np.sum(valid_mask)
                    print(f"Dataset: {name}, Number of valid genes: {num_valid_genes}")

    def __getitem__(self, idx):
        counts, idx, dataset, dataset_num = super().__getitem__(idx)
        if dataset in self.valid_gene_index and not utils.is_valid_uuid(dataset):
            valid_mask = self.valid_gene_index[dataset]
            counts = counts[:, valid_mask]
        return counts, idx, dataset, dataset_num

class VCIDatasetSentenceCollator(object):
    def __init__(self, cfg):
        self.pad_length = cfg.dataset.pad_length
        self.P = cfg.dataset.P
        self.N = cfg.dataset.N
        self.cfg = cfg

        self.dataset_to_protein_embeddings = torch.load(
            self.cfg.dataset.protein_emb_file_format.format(
                self.cfg.tokenizer.token_dim
            )
        )

    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length))

        idxs = torch.zeros(batch_size)
        Xs = torch.zeros((batch_size, (self.P + self.N)))
        Ys = torch.zeros((batch_size, (self.P + self.N)))

        dataset_nums = torch.zeros(batch_size)

        largest_cnt = max([x[0].shape[1] for x in batch])
        batch_weights = torch.zeros((batch_size, largest_cnt))

        i = 0
        max_len = 0

        for counts, idx, dataset, dataset_num, gene_indices, gene_scores in batch:
            (bs, xx, yy, batch_weight) = self.sample_cell_sentences(counts, dataset, gene_indices, gene_scores)

            batch_sentences[i, :] = bs
            batch_weight = batch_weight.squeeze()
            batch_weights[i, :len(batch_weight)] = batch_weight

            max_len = max(max_len, self.cfg.dataset.pad_length)
            idxs[i] = idx

            Xs[i] = xx  # [pn_idx]
            Ys[i] = yy.squeeze()  # [pn_idx]
            dataset_nums[i] = dataset_num
            i += 1

        return (
            batch_sentences[:, :max_len],
            Xs,
            Ys,
            idxs,
            batch_weights
        )

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sample_cell_sentences(self, counts, dataset, gene_indices, gene_scores):
        if torch.isnan(counts).any():
            log.error(f"NaN values in counts for dataset {dataset}")
        expression_weights = torch.log1p(counts)
        expression_weights = (expression_weights / torch.sum(expression_weights))

        ds_emb_idxs = self.dataset_to_protein_embeddings[dataset]
        cell_sentences = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        task_counts = torch.zeros((counts.shape[0], self.cfg.dataset.P + self.cfg.dataset.N))
        task_sentence = torch.zeros((counts.shape[0], self.cfg.dataset.P + self.cfg.dataset.N))

        # Available length after CLS token
        available_length = self.cfg.dataset.pad_length - 1
        half_len = available_length

        for c, cell in enumerate(counts):
            genes_ranked_exp = torch.argsort(cell, descending=True)[:half_len]

            # sample_size = (half_len - genes_ranked_exp.shape[0]) + half_len + 1
            # gened_sampled_by_exp = torch.multinomial(torch.nn.functional.softmax(expression_weights[c]),
            #                                          sample_size, replacement=True)

            # Combine into final sequence
            cell_sentences[c, 0] = self.cfg.dataset.cls_token_idx
            cell_sentences[c, 1: genes_ranked_exp.shape[0] + 1] = genes_ranked_exp

            # Convert tokens to Embeddings
            # this also includes the cls token, but we will override it later with a learnable torch vector
            # that logic is in model.py _compute_embedding_for_batch
            cell_sentences[c, :] = ds_emb_idxs[cell_sentences[c, :].to(torch.int32)]

            de_budget = self.cfg.dataset.P // 2
            replacement=False
            if gene_indices.shape[0] < de_budget:
                replacement=True
            task_sentence[c, :de_budget] = gene_indices[torch.multinomial(gene_scores, de_budget, replacement=replacement)]

            exp_genes = cell[cell > 0]
            unexp_genes = cell[cell < 1]

            if len(exp_genes) > de_budget:
                task_sentence[c, de_budget:self.cfg.dataset.P] = torch.randperm(len(exp_genes)) [0:de_budget]
            else:
                task_sentence[c, de_budget:self.cfg.dataset.P] = torch.randint(len(exp_genes), (de_budget,))

            if len(unexp_genes) > self.cfg.dataset.N:
                task_sentence[c, self.cfg.dataset.P:] = torch.randperm(len(unexp_genes)) [0:self.cfg.dataset.N]
            else:
                task_sentence[c, self.cfg.dataset.P:] = torch.randint(len(unexp_genes), (self.cfg.dataset.N,))

            task_counts[c] = cell[task_sentence[c].to(torch.int32)]
            task_counts[c] = torch.nn.functional.normalize(task_counts[c], dim=0)
        return cell_sentences, task_sentence, task_counts, expression_weights
