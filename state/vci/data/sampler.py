import random
from torch.utils.data import BatchSampler

class DatasetBatchSampler(BatchSampler):
    """
    A BatchSampler that yields batches containing only cells
    from a single dataset at a time.
    """
    def __init__(self, dataset, batch_size, shuffle=True, drop_last=False):
        # We don't actually use `sampler`, but BatchSampler API requires it.
        super().__init__(sampler=list(range(len(dataset))), 
                         batch_size=batch_size,
                         drop_last=drop_last)

        if 'datasets' not in dir(dataset):
            # distributed sampler for some reason reinitializes this with a wrapper
            dataset = dataset.dataset
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        # Precompute global‐index ranges for each dataset
        offset = 0
        self.ranges = []

        for name in dataset.datasets:
            n = dataset.num_cells[name]
            self.ranges.append((name, offset, offset + n))
            offset += n

    def __iter__(self):
        per_ds = []
        for name, lo, hi in self.ranges:
            idxs = list(range(lo, hi))
            if self.shuffle:
                random.shuffle(idxs)
            per_ds.append(idxs)

        ds_order = list(range(len(per_ds)))
        if self.shuffle:
            random.shuffle(ds_order)

        for ds_i in ds_order:
            idxs = per_ds[ds_i]
            for i in range(0, len(idxs), self.batch_size):
                batch = idxs[i:i+self.batch_size]
                if len(batch) == self.batch_size or not self.drop_last:
                    yield batch

    def __len__(self):
        total = 0
        for _, lo, hi in self.ranges:
            n = hi - lo
            cnt = n // self.batch_size + (0 if self.drop_last or n % self.batch_size == 0 else 1)
            total += cnt
        return total