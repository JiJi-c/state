import os
import sys
import time
import logging
import pandas as pd

from random import randrange
from hydra import compose, initialize
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vci.data import H5adDatasetSentences, VCIDatasetSentenceCollator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tests")


def test_index_compute():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="defaults")
        df = pd.read_csv(cfg.dataset.path)
        num_cell_idx = df.num_cells.cumsum()
        file_cnt = num_cell_idx.shape[0]

        ds = H5adDatasetSentences(cfg)
        test_cnt = 10
        for i in range(test_cnt):
            ds_num = randrange(0, file_cnt)
            if ds_num == 0:
                lower_bound = 0
            else:
                lower_bound = num_cell_idx[ds_num - 1]
            upper_bond = num_cell_idx[ds_num]

            test_idx = randrange(lower_bound, upper_bond)
            expected_file_index = test_idx - lower_bound
            expected_ds_num = ds_num

            act_ds_name, act_file_index = ds._compute_index(test_idx)
            act_ds_num = ds.datasets_to_num[act_ds_name]
            assert expected_file_index == act_file_index
            assert expected_ds_num == act_ds_num


def test_data_index():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="defaults")

        t_0 = time.time()
        dataset = H5adDatasetSentences(cfg)
        multi_ds_sent_collator = VCIDatasetSentenceCollator(cfg)

        # Make the dataloader outside of the
        dataloader = DataLoader(dataset,
                                batch_size=cfg.model.batch_size,
                                shuffle=False,
                                collate_fn=multi_ds_sent_collator,
                                num_workers=3,
                                persistent_workers=True)

        t_1 = time.time()
        log.info(f'Time to load dataset {t_1 - t_0}')

        for i, batch in enumerate(dataloader):
            log.info(f'new_batch_{i}.pickle')
            assert batch[0].shape == (cfg.model.batch_size, cfg.dataset.pad_length)
            assert batch[1].shape == (cfg.model.batch_size, cfg.dataset.pad_length)

            # assert batch[2].shape == (cfg.model.batch_size, cfg.dataset.pad_length)
            # assert batch[3].shape == (cfg.model.batch_size, cfg.dataset.pad_length)

            # assert batch[4].shape == (cfg.model.batch_size,)
            # assert batch[5].shape == (cfg.model.batch_size,)
            # import pickle
            # with open(f'new_batch_{i}.pickle', 'wb') as handle:
            #     pickle.dump(batch, handle)
            if i > 100:
                break

        log.info(f'Time to iterate thru datset {time.time() - t_1}')
