import os
import h5py
import logging
import torch
import torch.utils.data as data
import torch.nn.functional as F
import functools
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

                gene_indices, gene_scores = None, None
                if self.cfg.experiment.deaware:
                    gene_indices, gene_scores = self._get_DE_scores(h5f, ds_idx, self.dataset_group_map[dataset])

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

    def __getitem__(self, idx):
        counts, idx, dataset, dataset_num, gene_indices, gene_scores = super().__getitem__(idx)
        if dataset in self.valid_gene_index and not utils.is_valid_uuid(dataset):
            valid_mask = self.valid_gene_index[dataset]
            counts = counts[:, valid_mask]
        return counts, idx, dataset, dataset_num, gene_indices, gene_scores

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
        masks = torch.zeros((batch_size, self.pad_length))

        dataset_nums = torch.zeros(batch_size)

        largest_cnt = max([x[0].shape[1] for x in batch])
        batch_weights = torch.zeros((batch_size, largest_cnt))

        total_counts_all = None
        if self.cfg.model.rda:
            total_counts_all = torch.zeros(batch_size)

        i = 0
        max_len = 0

        for counts, idx, dataset, dataset_num, gene_indices, gene_scores in batch:
            (bs, xx, yy, batch_weight, mask, cell_total_counts) = self.sample_cell_sentences(counts, dataset, gene_indices, gene_scores)

            batch_sentences[i, :] = bs
            masks[i, :] = mask
            batch_weight = batch_weight.squeeze()
            batch_weights[i, :len(batch_weight)] = batch_weight

            max_len = max(max_len, self.cfg.dataset.pad_length)
            idxs[i] = idx

            Xs[i] = xx  # [pn_idx]
            Ys[i] = yy.squeeze()  # [pn_idx]
            dataset_nums[i] = dataset_num
            if self.cfg.model.rda and cell_total_counts is not None:
                total_counts_all[i] = cell_total_counts[0]
            i += 1

        return (
            batch_sentences[:, :max_len],
            Xs,
            Ys,
            idxs,
            batch_weights,
            masks,
            total_counts_all if self.cfg.model.rda else None,
        )

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sample_cell_sentences(self, counts, dataset, gene_indices, gene_scores):
        if torch.isnan(counts).any():
            log.error(f"NaN values in counts for dataset {dataset}")

        if torch.any(counts < 0):
            counts = F.relu(counts)

        # if the data has not already been log transformed
        if torch.max(counts) > 20:
            expression_weights = torch.log1p(counts)
        else:
            expression_weights = counts
        expression_weights = (expression_weights / torch.sum(expression_weights))

        ds_emb_idxs = self.dataset_to_protein_embeddings[dataset]
        cell_sentences = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        task_counts = torch.zeros((counts.shape[0], self.cfg.dataset.P + self.cfg.dataset.N))
        task_sentence = torch.zeros((counts.shape[0], self.cfg.dataset.P + self.cfg.dataset.N))
        mask = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length), dtype=torch.bool)

        if self.cfg.model.rda:
            cell_total_counts = torch.zeros((counts.shape[0],))
        else:
            cell_total_counts = None

        for c, cell in enumerate(counts):
            if self.cfg.model.rda:
                total_cnt = torch.sum(cell)
                # if total_cnt <= 1:
                #     min_value = cell[cell > 0].min()
                #     max_value = cell.max()
                #     total_cnt = cell * (max_value - min_value) + min_value
                cell_total_counts[c] = total_cnt

            num_pos_genes = torch.sum(cell > 0)
            # this is either the number of positive genes, or the first pad_length / 2 most expressed genes
            # the first is only used if you have more expressed genes than pad_length / 2
            start_sentence = min(((self.cfg.dataset.pad_length - 1) // 2), num_pos_genes)
            genes_ranked_exp = torch.argsort(cell, descending=True)

            # Combine into final sequence
            cell_sentences[c, 0] = self.cfg.dataset.cls_token_idx
            # paste the most expressed genes first
            cell_sentences[c, 1: start_sentence + 1] = genes_ranked_exp[:start_sentence]
            # sample with replacement weighted by normalized log counts
            cell_sentences[c, start_sentence + 1:] = torch.multinomial(
                    expression_weights, self.cfg.dataset.pad_length - start_sentence - 1, replacement=True)

            # Convert tokens to Embeddings
            # this also includes the cls token, but we will override it later with a learnable torch vector
            # that logic is in model.py _compute_embedding_for_batch
            cell_sentences[c, :] = ds_emb_idxs[cell_sentences[c, :].to(torch.int32)]

            if gene_indices is not None:
                # Task sentence for DE aware
                de_budget = min(self.cfg.dataset.P // 4, len(gene_indices))

                # TODO: if this overlaps too much we might mask out too much of the sentence
                task_sentence[c, :de_budget] = gene_indices[torch.multinomial(gene_scores, de_budget, replacement=False)]

                # NOTE: this may overlap with the first half of task sentence. let's talk about this more.
                exp_genes = torch.where(cell > 0)[0]

                if len(exp_genes) > self.cfg.dataset.P - de_budget:
                    task_sentence[c, de_budget:self.cfg.dataset.P] = \
                        exp_genes[torch.randperm(len(exp_genes))[0:self.cfg.dataset.P - de_budget]]
                else:
                    # sample with replacement
                    task_sentence[c, de_budget:self.cfg.dataset.P] = \
                        exp_genes[torch.randint(len(exp_genes), (self.cfg.dataset.P - de_budget,))]

            else:
                exp_genes = torch.where(cell > 0)[0]
                if len(exp_genes) > self.cfg.dataset.P:
                    task_sentence[c, :self.cfg.dataset.P] = exp_genes[torch.randperm(len(exp_genes))[0:self.cfg.dataset.P]]
                else:
                    task_sentence[c, :self.cfg.dataset.P] = exp_genes[torch.randint(len(exp_genes), (self.cfg.dataset.P,))]

            unexp_genes = torch.where(cell < 1)[0]
            if len(unexp_genes) > self.cfg.dataset.N:
                task_sentence[c, self.cfg.dataset.P:] = unexp_genes[torch.randperm(len(unexp_genes)) [0:self.cfg.dataset.N]]
            else:
                task_sentence[c, self.cfg.dataset.P:] = unexp_genes[torch.randint(len(unexp_genes), (self.cfg.dataset.N,))]

            task_counts[c] = cell[task_sentence[c].to(torch.int32)]

            if self.cfg.loss.name == "cross_entropy":
                # binarize the counts to 0/1
                task_counts[c] = (task_counts[c] > 0).float()
            else:
                # normalize the counts to sum to 1
                task_counts[c] = torch.nn.functional.normalize(task_counts[c], dim=0)

            # convert from dataset specific gene indices to global gene indices
            task_sentence[c: ] = ds_emb_idxs[task_sentence[c, :].to(torch.int32)]

            # mask out the task genes from the cell sentence
            task_gene_set = torch.tensor(task_sentence[c].tolist(), dtype=cell_sentences.dtype)
            potential_mask = torch.isin(cell_sentences[c], task_gene_set)

            # Calculate target number of masked tokens 
            target_mask_count = int(self.cfg.task.mask * self.cfg.dataset.pad_length)
            current_mask_count = potential_mask.sum().item()

            if current_mask_count > target_mask_count:
                # Too many tokens are being masked - randomly select subset
                # Only consider indices after the CLS token (index 0)
                mask_indices = torch.where(potential_mask[1:])[0] + 1  # +1 to adjust for offset
                keep_indices = torch.randperm(len(mask_indices))[:target_mask_count]
                selected_indices = mask_indices[keep_indices]
                
                # Create new mask with only the selected indices, ensuring CLS is not masked
                final_mask = torch.zeros_like(potential_mask)
                final_mask[selected_indices] = True
                mask[c] = final_mask
            elif current_mask_count < target_mask_count:
                # Not enough tokens masked - we need to mask additional tokens
                non_masked = ~potential_mask

                # Exclude the CLS token (index 0) by only considering indices 1 and up
                non_masked_indices = torch.where(non_masked[1:])[0] + 1  # +1 to adjust for offset
                
                # Calculate how many more tokens to mask
                additional_needed = target_mask_count - current_mask_count
                additional_needed = min(additional_needed, len(non_masked_indices))
                
                if len(non_masked_indices) > 0 and additional_needed > 0:
                    additional_indices = non_masked_indices[torch.randperm(len(non_masked_indices))[:additional_needed]]
                    potential_mask[additional_indices] = True
                
                mask[c] = potential_mask
            else:
                # Exactly self.cfg.task.mask percent are masked, use the potential mask as is
                mask[c] = potential_mask

            # make sure that the CLS token is never masked out.
            mask[c, 0] = False

        return cell_sentences, task_sentence, task_counts, counts, mask, cell_total_counts if self.cfg.model.rda else None
