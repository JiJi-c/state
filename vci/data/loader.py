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
                      workers=1,
                      data_dir=None,
                      datasets=None,
                      shape_dict=None,
                      adata=None,
                      adata_name=None,
                      shuffle=False,
                      sentence_collator=None):
        '''
        Expected to be used for inference
        Either datasets and shape_dict or adata and adata_name should be provided
        '''
        if datasets is None and adata is None:
            raise ValueError('Either datasets and shape_dict or adata and adata_name should be provided')

        if adata is not None:
            shuffle = False

        if data_dir:
            utils.get_dataset_cfg(cfg).data_dir = data_dir

        dataset = H5adSentenceDataset(cfg,
                                      datasets=datasets,
                                      shape_dict=shape_dict,
                                      adata=adata,
                                      adata_name=adata_name)
        if sentence_collator is None:
            sentence_collator = VCIDatasetSentenceCollator(cfg)
        dataloader = DataLoader(dataset,
                                batch_size=cfg.model.batch_size,
                                shuffle=shuffle,
                                collate_fn=sentence_collator,
                                num_workers=workers,
                                persistent_workers=True)
        return dataloader


class NpzMultiDataset(data.Dataset):
    def __init__(self, cfg, test=False) -> None:
        super(NpzMultiDataset, self).__init__()

        self.cfg = cfg
        ds_path = utils.get_dataset_cfg(cfg).train
        if test:
            ds_path = utils.get_dataset_cfg(cfg).val

        _, self.datasets, self.shapes_dict, self.dataset_path_map, self.dataset_group_map = utils.get_shapes_dict(ds_path)

        self.num_cells = {}
        self.num_genes = {}

        self.total_num_cells = 0
        for name in self.datasets:
            num_cells, num_genes = self.shapes_dict[name]
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes

            self.total_num_cells += num_cells

        self.datasets_to_num = {k: v for k, v in zip(self.datasets, range(len(self.datasets)))}


    @functools.lru_cache
    def dataset_file(self, dataset):
        cts = np.memmap(f"/large_experiments/ctc/ML/data/cell/observational/" + f"{dataset}_counts.npz",
                            dtype='int64', mode='r', shape=self.shapes_dict[dataset])
        return cts

    def __getitem__(self, idx):
        if isinstance(idx, int):
            for dataset in sorted(self.datasets):
                if idx < self.num_cells[dataset]:
                    cts = self.dataset_file(dataset)
                    counts = cts[idx]
                    counts = np.ascontiguousarray(counts)
                    counts = torch.tensor(counts).unsqueeze(0)
                    dataset_num = self.datasets_to_num[dataset]
                    return counts, idx, dataset, dataset_num,  None, None
                else:
                    idx -= self.num_cells[dataset]
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


class H5adSentenceDataset(data.Dataset):
    def __init__(self,
                 cfg,
                 test=False,
                 datasets=None,
                 shape_dict=None,
                 adata=None,
                 adata_name=None) -> None:
        super(H5adSentenceDataset, self).__init__()

        self.adata = None
        self.adata_name = adata_name
        self.test = test
        if adata is not None:
            self.adata = adata
            self.datasets = [adata_name]
            self.shapes_dict = {self.datasets[0]: adata.shape}
        elif datasets is None:
            ds_path = utils.get_dataset_cfg(cfg).train
            if test:
                ds_path = utils.get_dataset_cfg(cfg).val
            _, self.datasets, self.shapes_dict, self.dataset_path_map, self.dataset_group_map = \
                utils.get_shapes_dict(ds_path, utils.get_dataset_cfg(cfg).get('filter_by_species'))
        else:
            assert shape_dict is not None
            assert len(datasets) == len(shape_dict)
            self.datasets = datasets
            self.shapes_dict = shape_dict
            self.dataset_path_map = {dataset: dataset for dataset in datasets}

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
        return gene_indices, gene_scores

    def __getitem__(self, idx):
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

        except Exception as iex:
            log.exception(f"Error in dataset {dataset} at index {ds_idx}")
            raise iex

        dataset_num = self.datasets_to_num[dataset]
        return counts, idx, dataset, dataset_num, gene_indices, gene_scores

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


class GeneFilterDataset(H5adSentenceDataset):
    def __init__(self, cfg, test=False, datasets=None, shape_dict=None, adata=None, adata_name=None) -> None:
        super(GeneFilterDataset, self).__init__(cfg, test, datasets, shape_dict, adata, adata_name)
        self.valid_gene_index = {}
        if utils.get_embedding_cfg(cfg).valid_genes_masks is not None:
            self.valid_gene_index = torch.load(utils.get_embedding_cfg(cfg).valid_genes_masks)
        elif utils.get_embedding_cfg(self.cfg).ds_emb_mapping is not None:
            esm_data = torch.load(utils.get_embedding_cfg(self.cfg).all_embeddings)
            valid_genes_list = list(esm_data.keys())
            for name in self.datasets:
                if not utils.is_valid_uuid(name): # had to add this in for now as cellxgene h5ad fles don't have gene_name object but tahoe does
                    a = self.dataset_file(name)
                    gene_names = np.array([g.decode('utf-8') for g in a["/var/gene_name"][:]])  # Decode byte strings
                    valid_mask = np.isin(gene_names, valid_genes_list)
                    self.valid_gene_index[name] = valid_mask

    def __getitem__(self, idx):
        counts, idx, dataset, dataset_num, gene_indices, gene_scores = super().__getitem__(idx)
        if dataset in self.valid_gene_index:
            valid_mask = self.valid_gene_index[dataset]
            counts = counts[:, valid_mask]
        return counts, idx, dataset, dataset_num, gene_indices, gene_scores


class VCIDatasetSentenceCollator(object):
    def __init__(self, cfg):
        self.pad_length = cfg.dataset.pad_length
        self.P = cfg.dataset.P
        self.N = cfg.dataset.N
        self.S = cfg.dataset.S
        self.cfg = cfg

        self.dataset_to_protein_embeddings = torch.load(
            utils.get_embedding_cfg(self.cfg).ds_emb_mapping.format(
                utils.get_embedding_cfg(self.cfg).size
            )
        )

        self.global_to_local = {}
        for dataset_name, ds_emb_idxs in self.dataset_to_protein_embeddings.items():
            # Create a tensor filled with -1 (indicating not present in this dataset)
            reverse_mapping = torch.full((19790,), -1, dtype=torch.int64)
            
            local_indices = torch.arange(ds_emb_idxs.size(0), dtype=torch.int64)
            mask = (ds_emb_idxs >= 0) & (ds_emb_idxs < 19790)
            reverse_mapping[ds_emb_idxs[mask]] = local_indices[mask]
            self.global_to_local[dataset_name] = reverse_mapping

    def __call__(self, batch):
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length), dtype=torch.int32)
        batch_sentences_counts = torch.zeros((batch_size, self.pad_length))
        masks = torch.zeros((batch_size, self.pad_length), dtype=torch.bool)

        idxs = torch.zeros(batch_size, dtype=torch.int32)
        if self.cfg.loss.name == "tabular":
            task_num = self.P + self.N + self.S
        else:
            task_num = self.P + self.N
        Xs = torch.zeros((batch_size, (task_num)), dtype=torch.int32)
        Ys = torch.zeros((batch_size, (task_num)))

        dataset_nums = torch.zeros(batch_size, dtype=torch.int32)

        largest_cnt = max([x[0].shape[1] for x in batch])
        batch_weights = torch.zeros((batch_size, largest_cnt))

        total_counts_all = None
        if self.cfg.model.rda:
            total_counts_all = torch.zeros(batch_size)

        if self.cfg.loss.name == "tabular":
            shared_genes = torch.randint(low=0, high=19789, size=(self.S,), dtype=torch.int32)
        else:
            shared_genes = None

        i = 0
        max_len = 0
        datasets = []
        for counts, idx, dataset, dataset_num, gene_indices, gene_scores in batch:
            (bs, xx, yy, batch_weight, mask, cell_total_counts, cell_sentence_counts) = self.sample_cell_sentences(counts, dataset, gene_indices, gene_scores, shared_genes)
            datasets.append(dataset)

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
            if self.cfg.model.counts and cell_sentence_counts is not None:
                batch_sentences_counts[i, :] = cell_sentence_counts
            i += 1

        return (
            batch_sentences[:, :max_len],
            Xs,
            Ys,
            idxs,
            batch_weights,
            masks,
            total_counts_all if self.cfg.model.rda else None,
            batch_sentences_counts if self.cfg.model.counts else None,
        )

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # sampling a single cell sentence
    def sample_cell_sentences(self, counts, dataset, gene_indices, gene_scores, shared_genes=None):
        if torch.isnan(counts).any():
            log.error(f"NaN values in counts for dataset {dataset}")

        if torch.any(counts < 0):
            counts = F.relu(counts)

        # if the data has not already been log transformed
        if torch.max(counts) > 20: # CAN WE CHANGE THIS TO INT VS REAL
            counts = torch.log1p(counts)

        if counts.sum() == 0:
            expression_weights = F.softmax(counts, dim=1)
        else:
            expression_weights = counts / torch.sum(counts, dim=1, keepdim=True)

        ds_emb_idxs = self.dataset_to_protein_embeddings[dataset]
        if isinstance(ds_emb_idxs, np.ndarray):
            ds_emb_idxs = torch.tensor(ds_emb_idxs)

        cell_sentences = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        cell_sentence_counts = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        mask = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length), dtype=torch.bool)

        if self.cfg.loss.name == "tabular":
            # include capacity for shared genes
            task_num = self.cfg.dataset.P + self.cfg.dataset.N + self.cfg.dataset.S
        else:
            task_num = self.cfg.dataset.P + self.cfg.dataset.N

        task_counts = torch.zeros((counts.shape[0], task_num))
        task_sentence = torch.zeros((counts.shape[0], task_num))

        if self.cfg.model.rda:
            cell_total_counts = torch.zeros((counts.shape[0],))
        else:
            cell_total_counts = None

        # len(counts) = 1, e.g., we are looping over [cell]
        for c, cell in enumerate(counts):
            num_pos_genes = torch.sum(cell > 0)
            # this is either the number of positive genes, or the first pad_length / 2 most expressed genes
            # the first is only used if you have more expressed genes than pad_length / 2
            if self.cfg.model.counts:
                # shuffle before argsort - randomly break ties so we select random permuted genes
                indices = torch.randperm(cell.shape[-1])
                shuffled_cell = cell[indices]
                shuffled_genes_ranked_exp = torch.argsort(shuffled_cell, descending=True)
                genes_ranked_exp = indices[shuffled_genes_ranked_exp]
                cell_sentences[c, 0] = self.cfg.dataset.cls_token_idx
                if len(genes_ranked_exp) >= self.cfg.dataset.pad_length - 1:
                    cell_sentences[c, 1:] = genes_ranked_exp[:self.cfg.dataset.pad_length-1]
                else:
                    # take the nonzero genes first
                    num_nonzero = min(num_pos_genes, self.cfg.dataset.pad_length - 1)
                    cell_sentences[c, 1:num_nonzero+1] = genes_ranked_exp[:num_nonzero]

                    # sample the unexpressed genes with replacement
                    remaining_slots = self.cfg.dataset.pad_length - 1 - num_nonzero
                    unexpressed_genes = genes_ranked_exp[num_nonzero:]
                    cell_sentences[c, num_nonzero+1:] = unexpressed_genes[torch.randint(len(unexpressed_genes), (remaining_slots,))]
            else:
                start_sentence = min(((self.cfg.dataset.pad_length - 1) // 2), num_pos_genes)
                genes_ranked_exp = torch.argsort(cell, descending=True)

                # Combine into final sequence
                cell_sentences[c, 0] = self.cfg.dataset.cls_token_idx
                # paste the most expressed genes first
                cell_sentences[c, 1: start_sentence + 1] = genes_ranked_exp[:start_sentence]
                # sample with replacement weighted by normalized log counts
                cell_sentences[c, start_sentence + 1:] = torch.multinomial(
                        expression_weights, self.cfg.dataset.pad_length - start_sentence - 1, replacement=True)

            if self.cfg.model.counts:
                cell_sentence_counts[c, :] = 100 * expression_weights[c, cell_sentences[c, :].to(torch.int32)]

            # Convert tokens to Embeddings - local to global
            # this also includes the cls token, but we will override it later with a learnable torch vector
            cell_sentences[c, :] = ds_emb_idxs[cell_sentences[c, :].to(torch.int32)]

            # LOGIC FOR TASK SENTENCE
            if gene_indices is not None:
                # Task sentence for DE aware
                de_budget = min(self.cfg.dataset.P // 4, len(gene_indices))

                # TODO: if this overlaps too much we might mask out too much of the sentence
                # THIS CONTAINS EXPRESSED AND UNEXPRESSED GENES
                task_sentence[c, :de_budget] = gene_indices[torch.multinomial(gene_scores, de_budget, replacement=False)]

                # NOTE: this may overlap with the first half of task sentence. let's talk about this more.
                exp_genes = torch.where(cell > 0)[0]

                if len(exp_genes) > self.cfg.dataset.P - de_budget:
                    task_sentence[c, de_budget:self.cfg.dataset.P] = \
                        exp_genes[torch.randperm(len(exp_genes))[0:self.cfg.dataset.P - de_budget]]
                elif len(exp_genes) > 0:
                    # sample with replacement
                    task_sentence[c, de_budget:self.cfg.dataset.P] = \
                        exp_genes[torch.randint(len(exp_genes), (self.cfg.dataset.P - de_budget,))]

            else:
                exp_genes = torch.where(cell > 0)[0]
                if len(exp_genes) > self.cfg.dataset.P:
                    task_sentence[c, :self.cfg.dataset.P] = exp_genes[torch.randperm(len(exp_genes))[0:self.cfg.dataset.P]]
                elif len(exp_genes) > 0:
                    task_sentence[c, :self.cfg.dataset.P] = exp_genes[torch.randint(len(exp_genes), (self.cfg.dataset.P,))]

            # get the total number of genes unique to this cell; everything
            # past this are shared genes across all cells in a batch, used for tabular loss
            unshared_num = self.cfg.dataset.P + self.cfg.dataset.N

            # DE AWARE FOR UNEXPRESSED GENES???
            unexp_genes = torch.where(cell < 1)[0]
            if len(unexp_genes) > self.cfg.dataset.N:
                task_sentence[c, self.cfg.dataset.P:unshared_num] = unexp_genes[torch.randperm(len(unexp_genes)) [0:self.cfg.dataset.N]]
            else:
                task_sentence[c, self.cfg.dataset.P:unshared_num] = unexp_genes[torch.randint(len(unexp_genes), (self.cfg.dataset.N,))]

            # set counts for unshared genes
            task_counts[c, :unshared_num] = cell[task_sentence[c, :unshared_num].to(torch.int32)]

            # convert from dataset specific gene indices to global gene indices
            # only do this for everything up to shared genes, which are already global indices
            task_sentence[c, :unshared_num] = ds_emb_idxs[task_sentence[c, :unshared_num].to(torch.int32)]

            # now take care of shared genes across all cells in the batch
            if shared_genes is not None:
                # Overwrite the final positions of task_sentence (adjust as needed)
                task_sentence[c, unshared_num:] = shared_genes

                # convert the shared_genes, which are global indices, to the dataset specific indices
                local_indices = self.global_to_local[dataset][shared_genes].to(cell.device)
                shared_counts = torch.zeros(local_indices.shape, dtype=cell.dtype, device=cell.device)
                valid_mask = local_indices != -1
                if valid_mask.any():
                    shared_counts[valid_mask] = cell[local_indices[valid_mask]]

                # for indices which are -1, count is 0, else index into cell
                task_counts[c, unshared_num:] = shared_counts

            if self.cfg.model.rda:
                # sum the counts of the task sentence before converting to global indices
                cell_total_counts[c] = torch.sum(task_counts[c])

            if self.cfg.loss.name == "cross_entropy":
                # binarize the counts to 0/1
                task_counts[c] = (task_counts[c] > 0).float()

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

                # TODO: DOUBLE CHECK THE MASKING FOR CELL SENTENCE VS TASK SENTENCE.
                # housekeeping genes, ribosomal and mitochondrial genes take up a lot of the cell sentence
                # many of these genes are expressed in every cell - log transform helps a lot

                # 1. how much are we masking
                # 2. on average how much overlap is there
                # 3. entropy in the cell sentence and the task sentence

            # make sure that the CLS token is never masked out.
            mask[c, 0] = False

        return (
            cell_sentences,
            task_sentence,
            task_counts,
            counts,
            mask,
            cell_total_counts if self.cfg.model.rda else None,
            cell_sentence_counts if self.cfg.model.counts else None,
        )