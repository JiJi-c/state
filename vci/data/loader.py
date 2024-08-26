"""
Dataloaders

"""

import os
import h5py
import logging
import torch
import torch.utils.data as data

from typing import Dict

import vci.data.utils as utils


log = logging.getLogger(__file__)


class MultiDatasetSentences(data.Dataset):
    def __init__(self, cfg) -> None:
        super(MultiDatasetSentences, self).__init__()

        _, self.datasets, self.shapes_dict = utils.get_shapes_dict(cfg.dataset.path)
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

    def __getitem__(self, idx):
        if isinstance(idx, int):
            for dataset in sorted(self.datasets):
                if idx < self.num_cells[dataset]:
                    datafile = os.path.join(
                        f"{self.cfg.dataset.data_dir}", f"{dataset}.h5ad"
                    )
                    with h5py.File(datafile, "r") as h5f:
                        attrs = dict(h5f['X'].attrs)
                        if attrs['encoding-type'] == 'csr_matrix':
                            indptrs = h5f["/X/indptr"]
                            start_ptr = indptrs[idx]
                            end_ptr = indptrs[idx + 1]
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
                            counts = torch.tensor(h5f["X"][idx]).unsqueeze(0)

                    dataset_num = self.datasets_to_num[dataset]
                    return counts, idx, dataset, dataset_num
                else:
                    idx -= self.num_cells[dataset]
            raise IndexError
        else:
            raise NotImplementedError

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes


class MultiDatasetSentenceCollator(object):
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
            idxs,
            dataset_nums.long(),
        )

    def sample_cell_sentences_batched(self, batch):
        cnts = []
        for counts, idx, dataset, dataset_num in batch:
            cnts.append(counts)

        batch_weights = torch.log1p(counts)
        batch_weights = batch_weights / torch.sum(batch_weights)

        cell_sentences_pe, mask, cell_outputs_X_pe, cell_outputs_Y = None
        return (cell_sentences_pe, mask, cell_outputs_X_pe, cell_outputs_Y)


    def sample_cell_sentences(self, counts, dataset):

        dataset_idxs = self.dataset_to_protein_embeddings[dataset]
        cell_sentences = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length))
        mask = torch.zeros((counts.shape[0], self.cfg.dataset.pad_length), dtype=bool)

        cell_outputs_X = torch.zeros(
            (counts.shape[0], self.cfg.dataset.P + self.cfg.dataset.N)
        )
        cell_outputs_Y = torch.zeros(
            (counts.shape[0], self.cfg.dataset.P + self.cfg.dataset.N)
        )

        for c, cell in enumerate(counts):
            pos_genes = torch.where(counts[c] > 0)[0]
            neg_genes = torch.where(counts[c] < 1)[0]
            if len(pos_genes) == 0:
                pos_genes = neg_genes

            weights = torch.log1p(counts)[c]

            # drop these out
            mask_weights = torch.randperm(len(pos_genes))[:max(1, round(len(pos_genes) * 0.2))]
            mask_weights = pos_genes[mask_weights]
            weights[mask_weights] = 0
            weights[weights < 0] = 0
            # weights = weights / weights.sum()  # NORM after mask
            weights = torch.nan_to_num(weights, nan=0, neginf=0)

            weights = torch.nn.functional.softmax(weights, dim=0)
            choice_idx = torch.multinomial(weights,
                                            self.cfg.model.sample_size,
                                            replacement=True)

            ordered_choice_idx = torch.full((self.cfg.dataset.pad_length,),
                                         self.cfg.dataset.cls_token_idx)
            i = 1  # continue on to the rest of the sequence with left bracket being assumed.\

            ordered_choice_idx[i:(i + len(choice_idx))] = dataset_idxs[choice_idx]  # convert
            i += len(choice_idx)
            ordered_choice_idx[i:] = self.cfg.dataset.chrom_token_right_idx  # mask
            i += 1

            remainder_len = self.cfg.dataset.pad_length - i
            cell_mask = torch.concat(
                (
                    torch.zeros(i, dtype=bool),
                    # pay attention to all of these tokens, ignore the rest!
                    torch.ones(remainder_len, dtype=bool),
                )
            )

            mask[c, :] = cell_mask

            ordered_choice_idx[i:] = self.cfg.dataset.pad_token_idx  # mask

            cell_sentences[c, :] = ordered_choice_idx

            choice_idx_ouput_p = mask_weights  # use the masked genes as task
            if len(mask_weights) > self.cfg.dataset.P:
                # subset of masked genes
                choice_idx_ouput_p = mask_weights[\
                    torch.randperm(len(mask_weights))[:self.cfg.dataset.P]]
            elif len(mask_weights) < self.cfg.dataset.P:
                # remaining to be choosen
                choice_idx_ouput_p = torch.randint(len(mask_weights), (self.cfg.dataset.P,))

            if self.cfg.dataset.N > len(neg_genes):
                choice_idx_ouput_n = torch.randint(len(neg_genes), (self.cfg.dataset.N,))
            else:
                choice_idx_ouput_n = torch.randperm(len(neg_genes))[:self.cfg.dataset.N]
            choice_idx_ouput_n = neg_genes[choice_idx_ouput_n]

            cell_outputs_X[c] = torch.tensor(
                torch.cat((choice_idx_ouput_p, choice_idx_ouput_n))
            )
            cell_outputs_Y[c] = torch.cat(
                (torch.ones(self.cfg.dataset.P), torch.zeros(self.cfg.dataset.N))
            )

        cell_sentences_pe = cell_sentences.long()
        cell_outputs_X_pe = dataset_idxs[cell_outputs_X.long()]

        return (cell_sentences_pe, mask, cell_outputs_X_pe, cell_outputs_Y)

