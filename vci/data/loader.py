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

# THIS SHOULD ONLY BE USED FOR INFERENCE
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

        # THIS WAS PREVIOUSLY H5AD DATASET, but changed to
        # FilteredGenesCounts from previous commit to embed tahoe
        # Ideally we want to set this on the fly. is it safe to always be filtered?
        dataset = FilteredGenesCounts(cfg,
                                      datasets=datasets,
                                      shape_dict=shape_dict,
                                      adata=adata,
                                      adata_name=adata_name)
        if sentence_collator is None:
            sentence_collator = VCIDatasetSentenceCollator(cfg, valid_gene_mask=dataset.valid_gene_index)
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

    def __getitem__(self, idx):
        if self.adata is not None:
            # block is only used during validation
            # if .X is a numpy.ndarray
            if isinstance(self.adata.X, np.ndarray):
                counts = torch.tensor(self.adata.X[idx]).reshape(1, -1)
            else:
                counts = torch.tensor(self.adata.X[idx].todense())

            dataset = self.adata_name
            dataset_num = 0
            return counts, idx, dataset, dataset_num

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
                log.info(ds_idx)
                counts = torch.tensor(h5f["X"][ds_idx]).unsqueeze(0)

        except Exception as iex:
            log.exception(f"Error in dataset {dataset} at index {ds_idx}")
            raise iex

        dataset_num = self.datasets_to_num[dataset]
        return counts, idx, dataset, dataset_num

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes

class FilteredGenesCounts(H5adSentenceDataset):
    def __init__(self, cfg, test=False, datasets=None, shape_dict=None, adata=None, adata_name=None) -> None:
        super(FilteredGenesCounts, self).__init__(cfg, test, datasets, shape_dict, adata, adata_name)
        self.valid_gene_index = {}

        # make sure we get training datasets
        _, self.datasets, self.shapes_dict, self.dataset_path_map, self.dataset_group_map = utils.get_shapes_dict('/home/aadduri/state/h5ad_all.csv')

        # for inference, let's make sure this dataset's valid mask is available
        if adata_name is not None:
            # append it to self.datasets
            self.datasets.append(adata_name)
            self.shapes_dict[adata_name] = adata.shape

        if utils.get_embedding_cfg(self.cfg).ds_emb_mapping is not None:
            esm_data = torch.load(utils.get_embedding_cfg(self.cfg)['all_embeddings'])
            valid_genes_list = list(esm_data.keys())
            for name in self.datasets:
                if not utils.is_valid_uuid(name): # had to add this in for now as cellxgene h5ad fles don't have gene_name object but tahoe does
                    if adata is None:
                        a = self.dataset_file(name)
                        try:
                            gene_names = np.array([g.decode('utf-8') for g in a["/var/gene_name"][:]])  # Decode byte strings
                        except:
                            gene_categories = a["/var/gene_name/categories"][:]
                            gene_codes = np.array(a["/var/gene_name/codes"][:])
                            gene_names = np.array([g.decode('utf-8') for g in gene_categories[gene_codes]])
                        valid_mask = np.isin(gene_names, valid_genes_list)
                        self.valid_gene_index[name] = valid_mask
                    else:
                        gene_names = np.array(adata.var_names)
                        valid_mask = np.isin(gene_names, valid_genes_list)
                        self.valid_gene_index[name] = valid_mask

    def __getitem__(self, idx):
        counts, idx, dataset, dataset_num = super().__getitem__(idx)
        return counts, idx, dataset, dataset_num

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
        counts, idx, dataset, dataset_num = super().__getitem__(idx)
        return counts, idx, dataset, dataset_num

class VCIDatasetSentenceCollator(object):
    def __init__(self, cfg, valid_gene_mask=None):
        self.pad_length = cfg.dataset.pad_length
        self.P = cfg.dataset.P
        self.N = cfg.dataset.N
        self.S = cfg.dataset.S
        self.cfg = cfg

        # Load the dataset mappings
        self.use_dataset_info = getattr(cfg.model, "dataset_correction", False)
        self.batch_tabular_loss = getattr(cfg.model, "batch_tabular_loss", False)

        if valid_gene_mask is not None:
            # this branch is for inference
            self.valid_gene_mask = valid_gene_mask
        else:
            # otherwise for training, load from config
            gene_mask_file = utils.get_embedding_cfg(self.cfg).valid_genes_masks
            if gene_mask_file is not None:
                # we have a config for training
                self.valid_gene_mask = torch.load(gene_mask_file)
            else:
                # we don't have a config for training
                self.valid_gene_mask = None

        self.dataset_to_protein_embeddings = torch.load(
            utils.get_embedding_cfg(self.cfg).ds_emb_mapping.format(
                utils.get_embedding_cfg(self.cfg).size
            )
        )

        self.global_size = utils.get_embedding_cfg(self.cfg).num
        self.global_to_local = {}
        for dataset_name, ds_emb_idxs in self.dataset_to_protein_embeddings.items():
            # make sure tensor with long data type 
            ds_emb_idxs = torch.tensor(ds_emb_idxs, dtype=torch.long)
            # assert ds_emb_idxs.unique().numel() == ds_emb_idxs.numel(), f"duplicate global IDs in dataset {dataset_name}!"

            # Create a tensor filled with -1 (indicating not present in this dataset)
            reverse_mapping = torch.full((self.global_size,), -1, dtype=torch.int64)

            local_indices = torch.arange(ds_emb_idxs.size(0), dtype=torch.int64)
            mask = (ds_emb_idxs >= 0) & (ds_emb_idxs < self.global_size)
            reverse_mapping[ds_emb_idxs[mask]] = local_indices[mask]
            self.global_to_local[dataset_name] = reverse_mapping

        print(len(self.global_to_local))

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

        largest_cnt = max([x[0].shape[1] for x in batch])
        batch_weights = torch.zeros((batch_size, largest_cnt))

        total_counts_all = None
        if self.cfg.model.rda:
            total_counts_all = torch.zeros(batch_size)

        datasets = []
        for _, _, ds_name, _, in batch:
            datasets.append(ds_name)

        if self.cfg.loss.name == "tabular":
            if self.batch_tabular_loss:
                batch_ds = set(datasets)
                presence_masks = [ (self.global_to_local[ds] >= 0) for ds in batch_ds ]
                inter = presence_masks[0].clone()
                for m in presence_masks[1:]:
                    inter &= m
                candidates = torch.where(inter)[0]     # all global IDs present in every dataset
                n = candidates.numel()
                if n >= self.S:
                    # sample without replacement
                    idx = torch.randperm(n, device=candidates.device)[:self.S]
                    shared_genes = candidates[idx]
                elif n > 0:
                    # sample with replacement
                    idx = torch.randint(n, (self.S,), device=candidates.device)
                    shared_genes = candidates[idx]
                else:
                    # truly no overlap → random global pick
                    shared_genes = torch.randint(
                        low=0,
                        high=self.global_size,
                        size=(self.S,),
                        device=masks.device,
                        dtype=torch.long
                    )
            else:
                # Find genes shared across all datasets
                shared_mask = None
                for dataset in datasets:
                    dataset_mask = self.global_to_local[dataset] >= 0
                    if shared_mask is None:
                        shared_mask = dataset_mask
                    else:
                        shared_mask &= dataset_mask
                
                # Get indices of shared genes
                shared_indices = torch.where(shared_mask)[0]
                
                # Repeat shared genes to reach size S
                n_shared = shared_indices.size(0)
                if n_shared > 0:
                    # Calculate how many times to repeat and remainder
                    repeats = self.S // n_shared
                    remainder = self.S % n_shared
                    
                    # Repeat the full sequence
                    shared_genes = shared_indices.repeat(repeats)
                    
                    # Add remaining genes needed
                    if remainder > 0:
                        shared_genes = torch.cat([shared_genes, shared_indices[:remainder]])
                else:
                    # If no shared genes, sample randomly from global gene space
                    shared_genes = torch.randint(
                        low=0,
                        high=self.global_size,
                        size=(self.S,),
                        device=masks.device,
                        dtype=torch.long
                    )
        else:
            shared_genes = None

        dataset_nums = torch.zeros(batch_size, dtype=torch.int32)

        i = 0
        max_len = 0
        for counts, idx, dataset, dataset_num in batch:
            if self.valid_gene_mask is not None:
                if dataset in self.valid_gene_mask:
                    valid_mask = self.valid_gene_mask[dataset]
                else:
                    valid_mask = None
            else:
                valid_mask = None
            (bs, xx, yy, batch_weight, mask, cell_total_counts, cell_sentence_counts) = self.sample_cell_sentences(counts, dataset, shared_genes, valid_mask)

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
            dataset_nums if self.use_dataset_info else None,
        )

    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    # sampling a single cell sentence
    def sample_cell_sentences(self, counts_raw, dataset, shared_genes=None, valid_gene_mask=None):
        if torch.isnan(counts_raw).any():
            log.error(f"NaN values in counts for dataset {dataset}")

        ds_emb_idxs = torch.tensor(self.dataset_to_protein_embeddings[dataset], dtype=torch.long)
        counts = counts_raw
        if valid_gene_mask is not None:
            if ds_emb_idxs.shape[0] == valid_gene_mask.shape[0]:
                # IF THE MASK IS THE SAME SIZE AS THE DATASET
                # WITH -1 FOR MISSING GENES
                ds_emb_idxs = ds_emb_idxs[valid_gene_mask]
            else:
                # assert that valid_gene_mask.sum = ds_emb_idxs.shape[0]
                # OTHERWISE, DATASET COLUMNS HAVE ALREADY BEEN FILTERD
                assert valid_gene_mask.sum() == ds_emb_idxs.shape[0], f"Something wrong with filtering or mask for dataset {dataset}"
            if counts_raw.shape[1] == valid_gene_mask.shape[0]:
                # however, COUNTS ARE NEVER FILTERED
                counts = counts_raw[:, valid_gene_mask]

        if torch.any(counts < 0):
            counts = F.relu(counts)

        if torch.max(counts) > 35: # CAN WE CHANGE THIS TO INT VS REAL
            counts = torch.log1p(counts)

        if counts.sum() == 0:
            expression_weights = F.softmax(counts, dim=1)
        else:
            expression_weights = counts / torch.sum(counts, dim=1, keepdim=True)

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
            assert self.cfg.model.counts
            # shuffle before argsort - randomly break ties so we select random unexpressed genes each time, if pad_length > num_non_zero genes
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

            cell_sentence_counts[c, :] = 100 * expression_weights[c, cell_sentences[c, :].to(torch.int32)]

            # Convert tokens to Embeddings - local to global
            # this also includes the cls token, but we will override it later with a learnable torch vector
            cell_sentences[c, :] = ds_emb_idxs[cell_sentences[c, :].to(torch.int32)]

            # pick P expressed genes to mask for MLM
            exp_genes = torch.where(cell > 0)[0]
            if len(exp_genes) > self.cfg.dataset.P:
                task_sentence[c, :self.cfg.dataset.P] = exp_genes[torch.randperm(len(exp_genes))[0:self.cfg.dataset.P]]
            elif len(exp_genes) > 0:
                task_sentence[c, :self.cfg.dataset.P] = exp_genes[torch.randint(len(exp_genes), (self.cfg.dataset.P,))]

            # get the total number of genes unique to this cell; everything
            # past this are shared genes across all cells in a batch, used for tabular loss
            unshared_num = self.cfg.dataset.P + self.cfg.dataset.N

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
                # Overwrite the final positions of task_sentence 

                task_sentence[c, unshared_num:] = shared_genes # in the old impl these are global gene indices
                # task_sentence[c, unshared_num:] = ds_emb_idxs[shared_genes.to(torch.int32)] # in the new impl these are local gene indices

                # convert the shared_genes, which are global indices, to the dataset specific indices
                local_indices = self.global_to_local[dataset][shared_genes].to(cell.device) # in the old impl these are global gene indices
                # local_indices = shared_genes # in the new impl these are local gene indices

                shared_counts = torch.zeros(local_indices.shape, dtype=cell.dtype, device=cell.device)
                valid_mask = local_indices != -1
                if valid_mask.any():
                    shared_counts[valid_mask] = counts_raw[c, local_indices[valid_mask]]

                # for indices which are -1, count is 0, else index into cell
                task_counts[c, unshared_num:] = shared_counts

            assert self.cfg.model.rda
            # sum the counts of the task sentence
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

