import os
import h5py
import logging
import torch
import functools
import torch.utils.data as data
import numpy as np

from typing import Dict

import vci.utils as utils


log = logging.getLogger(__file__)


class H5adDatasetSentences(data.Dataset):
    def __init__(self, cfg, test=False, datasets=None, shape_dict=None) -> None:
        super(H5adDatasetSentences, self).__init__()

        if datasets is None:
            ds_path = cfg.dataset.train
            if test:
                ds_path = cfg.dataset.test
            _, self.datasets, self.shapes_dict = utils.get_shapes_dict(ds_path)
        else:
            assert shape_dict is not None
            assert len(datasets) == len(shape_dict)
            self.datasets = datasets
            self.shapes_dict = shape_dict

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
        for dataset in sorted(self.datasets):
            if idx < self.num_cells[dataset]:
                return dataset, idx
            else:
                idx -= self.num_cells[dataset]
        raise IndexError

    @functools.lru_cache
    def dataset_file(self, dataset):
        datafile = os.path.join(self.cfg.dataset.data_dir, f"{dataset}.h5ad")
        return h5py.File(datafile, "r")

    def __getitem__(self, idx):
        if isinstance(idx, int):
            dataset, ds_idx = self._compute_index(idx)
            h5f = self.dataset_file(dataset)
            attrs = dict(h5f['X'].attrs)
            try:
                if attrs['encoding-type'] == 'csr_matrix':
                    # num_genes = attrs['shape'][1]
                    indptrs = h5f["/X/indptr"]
                    start_ptr = indptrs[ds_idx]
                    end_ptr = indptrs[ds_idx + 1]
                    sub_data = torch.tensor(
                        h5f["/X/data"][start_ptr:end_ptr],
                        dtype=torch.int32)
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
            except IndexError as iex:
                log.exception(f"Error in dataset {dataset} at index {ds_idx}")
                raise iex

            dataset_num = self.datasets_to_num[dataset]
            return counts, idx, dataset, dataset_num
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


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
        mask = torch.zeros((batch_size, self.pad_length), dtype=bool)

        idxs = torch.zeros(batch_size)
        Xs = torch.zeros((batch_size, (self.P + self.N)))
        Ys = torch.zeros((batch_size, (self.P + self.N)))

        dataset_nums = torch.zeros(batch_size)

        # return self.sample_cell_sentences_batched(batch)

        i = 0
        max_len = 0

        for counts, idx, dataset, dataset_num in batch:
            (bs, msk, xx, yy) = self.sample_cell_sentences(counts, dataset)

            yy = yy.squeeze()
            batch_sentences[i, :] = bs

            max_len = max(max_len, self.cfg.dataset.pad_length)
            mask[i, :] = msk
            idxs[i] = idx

            Xs[i] = xx  # [pn_idx]
            Ys[i] = yy  # [pn_idx]
            dataset_nums[i] = dataset_num
            i += 1

        return (
            batch_sentences[:, :max_len],
            mask[:, :max_len],
            Xs,
            Ys,
            idxs
        )

    def sample_cell_sentences_batched(self, batch):
        cnts = []
        for counts, idx, dataset, dataset_num in batch:
            cnts.append(counts)

        batch_weights = torch.log1p(counts)
        batch_weights = batch_weights / torch.sum(batch_weights)

        cell_sentences_pe, mask, cell_outputs_X_pe, cell_outputs_Y = None
        return (cell_sentences_pe, mask, cell_outputs_X_pe, cell_outputs_Y)


    def softmax(self, x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def sample_cell_sentences(self, counts, dataset):
        batch_weights = torch.log1p(counts)
        batch_weights = (batch_weights / torch.sum(batch_weights))

        dataset_idxs = self.dataset_to_protein_embeddings[dataset]
        cell_sentences = torch.zeros((counts.shape[0],  self.cfg.dataset.pad_length))

        mask = torch.zeros((counts.shape[0],  self.cfg.dataset.pad_length), dtype=bool)

        cell_outputs_X = torch.zeros((counts.shape[0],  self.cfg.dataset.P +  self.cfg.dataset.N))
        cell_outputs_Y = torch.zeros((counts.shape[0],  self.cfg.dataset.P +  self.cfg.dataset.N))

        for c, cell in enumerate(counts):
            pos_genes = torch.where(counts[c] > 0)[0]
            neg_genes = torch.where(counts[c] < 1)[0]
            if len(pos_genes) == 0:
                pos_genes = neg_genes

            weights = batch_weights[c]
            # 20% random dropout
            mask_weights = torch.randperm(len(pos_genes))[:max(1, round(len(pos_genes) * 0.2))]
            mask_weights = pos_genes[mask_weights]

            weights[mask_weights] = 0
            weights[weights < 0] = 0
            weights = torch.nan_to_num(weights, nan=0, neginf=0)
            weights = torch.nn.functional.softmax(weights, dim=0)

            choice_idx = torch.multinomial(weights,
                                           self.cfg.dataset.pad_length - 1,
                                           replacement=True)

            ordered_choice_idx = torch.full((self.cfg.dataset.pad_length,),
                                            self.cfg.dataset.cls_token_idx)

            i = 1  # continue on to the rest of the sequence with left bracket being assumed.\
            ordered_choice_idx[i:(self.cfg.dataset.pad_length)] = dataset_idxs[choice_idx]
            i = i + self.cfg.dataset.pad_length - 1

            remainder_len = ( self.cfg.dataset.pad_length - i)

            cell_mask = torch.concat((torch.zeros(i, dtype=bool),
                                      torch.ones(remainder_len, dtype=bool)))

            mask[c, :] = cell_mask

            ordered_choice_idx[i:] =  self.cfg.dataset.pad_token_idx  # mask

            cell_sentences[c, :] = ordered_choice_idx
            choice_idx_ouput_p = mask_weights  # use the masked genes as task
            if len(choice_idx_ouput_p) >  self.cfg.dataset.P:
                choice_idx_ouput_p = mask_weights[\
                    torch.randperm(len(mask_weights))[:self.cfg.dataset.P]]
            elif len(choice_idx_ouput_p) <  self.cfg.dataset.P:
                remainder =  self.cfg.dataset.P - len(choice_idx_ouput_p)  # remaining to be choosen
                choice_idx_ouput_p = torch.cat((choice_idx_ouput_p,
                                                pos_genes[torch.randint(len(pos_genes), (remainder,))]))

            if  self.cfg.dataset.N <= len(neg_genes):
                choice_idx_ouput_n = torch.randperm(len(neg_genes))[:self.cfg.dataset.N]
            else:
                choice_idx_ouput_n = torch.randint(len(neg_genes), (self.cfg.dataset.N,))

            choice_idx_ouput_n = neg_genes[choice_idx_ouput_n]

            cell_outputs_X[c] = torch.tensor(
                np.concatenate((choice_idx_ouput_p, choice_idx_ouput_n)))
            cell_outputs_Y[c] = torch.cat((torch.ones( self.cfg.dataset.P), torch.zeros( self.cfg.dataset.N)))

        cell_sentences_pe = cell_sentences.long()  # .unsqueeze(2) # all_pe[cell_sentences.long(), :]
        cell_outputs_X_pe = dataset_idxs[
            cell_outputs_X.long()]  # .unsqueeze(2) # all_pe[dataset_idxs[cell_outputs_X.long()], :]

        return cell_sentences_pe, mask, cell_outputs_X_pe, cell_outputs_Y
