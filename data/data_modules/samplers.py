import numpy as np
from data.dataset.perturbation_dataset import PerturbationDataset
from data.utils.data_utils import H5MetadataCache
from typing import List, Iterator
from torch.utils.data import Sampler
import logging
import time

logger = logging.getLogger(__name__)

class PerturbationBatchSampler(Sampler):
    """
    Samples batches ensuring that cells in each batch share the same 
    (cell_type, perturbation) combination, using only H5 codes.
    
    Instead of grouping by cell type and perturbation names, this sampler
    groups based on integer codes stored in the H5 file (e.g. `cell_type_codes`
    and `pert_codes` in the H5MetadataCache). This avoids repeated string operations.
    """
    
    def __init__(self, dataset: "MetadataConcatDataset", batch_size: int, drop_last: bool = False, cell_sentence_len: int = 512, test: bool = False, use_batch: bool = False):
        logger.info("Creating perturbation batch sampler with metadata caching (using codes)...")
        start_time = time.time()

        # If the provided dataset has a `.data_source` attribute, use that.
        self.dataset = dataset.data_source if hasattr(dataset, "data_source") else dataset
        self.batch_size = batch_size
        self.test = test
        self.use_batch = use_batch

        if self.test and self.batch_size != 1:
            logger.warning('Batch size should be 1 for test mode. Setting batch size to 1.')
            self.batch_size = 1

        self.cell_sentence_len = cell_sentence_len
        self.drop_last = drop_last
        
        # Create caches for all unique H5 files.
        self.metadata_caches = {}
        for subset in self.dataset.datasets:
            base_dataset: PerturbationDataset = subset.dataset
            self.metadata_caches[base_dataset.h5_path] = base_dataset.metadata_cache
        
        # Create batches using the code-based grouping.
        self.sentences = self._create_sentences()
        sentence_lens = [len(sentence) for sentence in self.sentences]
        avg_num = np.mean(sentence_lens)
        std_num = np.std(sentence_lens)
        tot_num = np.sum(sentence_lens)
        logger.info(f"Total # cells {tot_num}. Cell set size mean / std before resampling: {avg_num:.2f} / {std_num:.2f}.")

        # combine sentences into batches that are flattened
        logger.info(f"Creating meta-batches with cell_sentence_len={cell_sentence_len}...")
        self.batches = self._create_batches()
        self.tot_num = tot_num

        end_time = time.time()
        logger.info(
            f"Sampler created with {len(self.batches)} batches in {end_time - start_time:.2f} seconds."
        )

    def _create_batches(self) -> List[List[int]]:
        """
        Combines existing batches into meta-batches of size batch_size * cell_sentence_len,
        sampling with replacement if needed to reach cell_sentence_len.
        """
        all_batches = []
        current_batch = []
        
        num_full = 0
        num_partial = 0
        for sentence in self.sentences:
            # If batch is smaller than cell_sentence_len, sample with replacement
            if len(sentence) < self.cell_sentence_len and not self.test:
                # during inference, don't sample by replacement
                sentence = list(np.random.choice(sentence, size=self.cell_sentence_len, replace=True))
                num_partial += 1
            else:
                assert len(sentence) == self.cell_sentence_len or self.test
                num_full += 1

            sentence_len = len(sentence) if self.test else self.cell_sentence_len

            if len(current_batch) + len(sentence) <= self.batch_size * sentence_len:
                current_batch.extend(sentence)
            else:
                if current_batch:  # Add the completed meta-batch
                    all_batches.append(current_batch)
                current_batch = sentence
        logger.info(f"Of all batches, {num_full} were full and {num_partial} were partial.")
                
        # Add the last meta-batch if it exists
        if current_batch:
            all_batches.append(current_batch)
            
        return all_batches

    def _process_subset(self, global_offset: int, subset: "Subset") -> List[List[int]]:
        """
        Process a single subset to create batches based on H5 codes.
        
        For each subset, the method:
          - Retrieves the subset indices.
          - Extracts the corresponding cell type and perturbation codes from the cache.
          - Constructs a structured array with two fields (cell, pert) so that unique
            (cell_type, perturbation) pairs can be identified using np.unique.
          - For each unique pair, shuffles the indices and splits them into batches.
        """
        base_dataset = subset.dataset
        indices = np.array(subset.indices)
        cache: H5MetadataCache = self.metadata_caches[base_dataset.h5_path]
        
        # Use codes directly rather than names.
        cell_codes = cache.cell_type_codes[indices]
        pert_codes = cache.pert_codes[indices]

        if 'use_batch' in self.__dict__ and self.use_batch:
            # If using batch, we need to use the batch codes instead of cell type codes.
            batch_codes = cache.batch_codes[indices]
            # Also get batch codes if grouping by batch is desired.
            batch_codes = cache.batch_codes[indices]
            dt = np.dtype([
                ('batch', batch_codes.dtype),
                ('cell', cell_codes.dtype),
                ('pert', pert_codes.dtype)
            ])
            groups = np.empty(len(indices), dtype=dt)
            groups['batch'] = batch_codes
            groups['cell'] = cell_codes
            groups['pert'] = pert_codes
        else:
            dt = np.dtype([
                ('cell', cell_codes.dtype),
                ('pert', pert_codes.dtype)
            ])
            groups = np.empty(len(indices), dtype=dt)
            groups['cell'] = cell_codes
            groups['pert'] = pert_codes
        
        # Create global indices (assuming that indices in each subset refer to a global concatenation).
        global_indices = np.arange(global_offset, global_offset + len(indices))
        
        subset_batches = []
        # Group by unique (cell, pert) pairs.
        for group_key in np.unique(groups):
            mask = (groups == group_key)
            group_indices = global_indices[mask]
            np.random.shuffle(group_indices)
            
            # Split the group indices into batches.
            for i in range(0, len(group_indices), self.cell_sentence_len):
                sentence = group_indices[i:i + self.cell_sentence_len].tolist()
                if len(sentence) < self.cell_sentence_len and self.drop_last:
                    continue
                subset_batches.append(sentence)
                
        return subset_batches

    def _create_sentences(self) -> List[List[int]]:
        """
        Process each subset sequentially (across all datasets) and combine the batches.
        """
        global_offset = 0
        all_batches = []
        for subset in self.dataset.datasets:
            subset_batches = self._process_subset(global_offset, subset)
            all_batches.extend(subset_batches)
            global_offset += len(subset)

        np.random.shuffle(all_batches)
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
        # Shuffle the order of batches each time we iterate.
        np.random.shuffle(self.batches)
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)

class EpochPerturbationBatchSampler(Sampler):
    """
    Samples batches ensuring that cells in each batch share the same 
    (cell_type, perturbation) combination, using only H5 codes.
    
    Instead of grouping by cell type and perturbation names, this sampler
    groups based on integer codes stored in the H5 file (e.g. `cell_type_codes`
    and `pert_codes` in the H5MetadataCache). This avoids repeated string operations.

    Recomputes sentence groupings each epoch to provide different cell combinations,
    while still ensuring that cells in each group share the same properties.
    """
    
    def __init__(self, dataset: "MetadataConcatDataset", batch_size: int, drop_last: bool = False, cell_sentence_len: int = 512, test: bool = False, use_batch: bool = False):
        logger.info("Creating perturbation batch sampler with metadata caching (using codes)...")
        start_time = time.time()

        # If the provided dataset has a `.data_source` attribute, use that.
        self.dataset = dataset.data_source if hasattr(dataset, "data_source") else dataset
        self.batch_size = batch_size
        self.test = test
        self.use_batch = use_batch
        self.random_seed = 42
        self.epoch_counter = 0

        if self.test and self.batch_size != 1:
            logger.warning('Batch size should be 1 for test mode. Setting batch size to 1.')
            self.batch_size = 1

        self.cell_sentence_len = cell_sentence_len
        self.drop_last = drop_last
        
        # Create caches for all unique H5 files.
        self.metadata_caches = {}
        for subset in self.dataset.datasets:
            base_dataset: PerturbationDataset = subset.dataset
            self.metadata_caches[base_dataset.h5_path] = base_dataset.metadata_cache
        
        # Create batches using the code-based grouping.
        # Count total cells for reporting
        self.tot_num = self._count_total_cells()
        
        # Don't create sentences and batches at initialization
        # They will be created in __iter__ for each epoch
        logger.info(f"Sampler initialized. Will create sentences with cell_sentence_len={cell_sentence_len} each epoch.")

        end_time = time.time()
        logger.info(
            f"Sampler created in {end_time - start_time:.2f} seconds."
        )

    def _count_total_cells(self) -> int:
        """Count the total number of cells across all datasets."""
        total = 0
        for subset in self.dataset.datasets:
            total += len(subset)
        return total

    def _create_batches(self, sentences: List[List[int]], epoch_seed: int) -> List[List[int]]:
        """
        Combines existing batches into meta-batches of size batch_size * cell_sentence_len,
        sampling with replacement if needed to reach cell_sentence_len.
        Now uses epoch_seed for consistent randomness within an epoch.
        """
        all_batches = []
        current_batch = []

        # Create epoch-specific RNG
        epoch_rng = np.random.RandomState(epoch_seed)
        
        num_full = 0
        num_partial = 0
        for sentence in sentences:
            # If batch is smaller than cell_sentence_len, sample with replacement
            if len(sentence) < self.cell_sentence_len and not self.test:
                # during inference, don't sample by replacement
                sentence = list(epoch_rng.choice(sentence, size=self.cell_sentence_len, replace=True))
                num_partial += 1
            else:
                assert len(sentence) == self.cell_sentence_len or self.test
                num_full += 1

            sentence_len = len(sentence) if self.test else self.cell_sentence_len

            if len(current_batch) + len(sentence) <= self.batch_size * sentence_len:
                current_batch.extend(sentence)
            else:
                if current_batch:  # Add the completed meta-batch
                    all_batches.append(current_batch)
                current_batch = sentence
        logger.info(f"Epoch batches: {num_full} full and {num_partial} partial.")

                
        # Add the last meta-batch if it exists
        if current_batch:
            all_batches.append(current_batch)
            
        return all_batches

    def _process_subset(self, global_offset: int, subset: "Subset", epoch_seed: int) -> List[List[int]]:
        """
        Process a single subset to create batches based on H5 codes.
        Now uses epoch_seed for consistent randomness within an epoch.
        
        For each subset, the method:
          - Retrieves the subset indices.
          - Extracts the corresponding cell type and perturbation codes from the cache.
          - Constructs a structured array with two fields (cell, pert) so that unique
            (cell_type, perturbation) pairs can be identified using np.unique.
          - For each unique pair, shuffles the indices and splits them into batches.
        """
        base_dataset = subset.dataset
        indices = np.array(subset.indices)
        cache: H5MetadataCache = self.metadata_caches[base_dataset.h5_path]
        
        # Use codes directly rather than names.
        cell_codes = cache.cell_type_codes[indices]
        pert_codes = cache.pert_codes[indices]

        if 'use_batch' in self.__dict__ and self.use_batch:
            # If using batch, we need to use the batch codes instead of cell type codes.
            batch_codes = cache.batch_codes[indices]
            # Also get batch codes if grouping by batch is desired.
            batch_codes = cache.batch_codes[indices]
            dt = np.dtype([
                ('batch', batch_codes.dtype),
                ('cell', cell_codes.dtype),
                ('pert', pert_codes.dtype)
            ])
            groups = np.empty(len(indices), dtype=dt)
            groups['batch'] = batch_codes
            groups['cell'] = cell_codes
            groups['pert'] = pert_codes
        else:
            dt = np.dtype([
                ('cell', cell_codes.dtype),
                ('pert', pert_codes.dtype)
            ])
            groups = np.empty(len(indices), dtype=dt)
            groups['cell'] = cell_codes
            groups['pert'] = pert_codes
        
        # Create global indices (assuming that indices in each subset refer to a global concatenation).
        global_indices = np.arange(global_offset, global_offset + len(indices))

        epoch_rng = np.random.RandomState(epoch_seed)
        
        subset_batches = []
        # Group by unique (cell, pert) pairs.
        for group_key in np.unique(groups):
            mask = (groups == group_key)
            group_indices = global_indices[mask]
            epoch_rng.shuffle(group_indices)
            
            # Split the group indices into batches.
            for i in range(0, len(group_indices), self.cell_sentence_len):
                sentence = group_indices[i:i + self.cell_sentence_len].tolist()
                if len(sentence) < self.cell_sentence_len and self.drop_last:
                    continue
                subset_batches.append(sentence)
                
        return subset_batches

    def _create_sentences(self, epoch_seed: int) -> List[List[int]]:
        """
        Process each subset sequentially (across all datasets) and combine the batches.
        Now uses epoch_seed for consistent randomness within an epoch.
        """
        global_offset = 0
        all_batches = []
        for subset in self.dataset.datasets:
            subset_batches = self._process_subset(global_offset, subset, epoch_seed)
            all_batches.extend(subset_batches)
            global_offset += len(subset)

        # Use epoch-specific RNG for shuffling
        epoch_rng = np.random.RandomState(epoch_seed)
        epoch_rng.shuffle(all_batches)
        return all_batches

    def __iter__(self) -> Iterator[List[int]]:
       """
       Recomputes all sentences and batches each time we iterate.
       Uses a combination of self.random_seed and epoch_counter to ensure
       different random shuffling each epoch but reproducibility across runs.
       """
       # Calculate epoch seed by combining base seed with current epoch counter
       epoch_seed = self.random_seed + self.epoch_counter
       
       # Create new sentences with this epoch's seed
       sentences = self._create_sentences(epoch_seed)
       
       # Create batches for this epoch
       batches = self._create_batches(sentences, epoch_seed)
       
       # Shuffle batches with the epoch-specific seed
       epoch_rng = np.random.RandomState(epoch_seed)
       epoch_rng.shuffle(batches)
       
       # Increment epoch counter for next time
       self.epoch_counter += 1
       
       yield from batches

    def __len__(self) -> int:
        """
        Estimate the number of batches.
        Since we recompute batches each epoch, this is an approximation based on the initial seed.
        """
        # Create temporary sentences and batches to estimate batch count
        temp_sentences = self._create_sentences(self.random_seed)
        temp_batches = self._create_batches(temp_sentences, self.random_seed)
        return len(temp_batches)