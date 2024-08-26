import os
import sys
import time
import logging

from hydra import compose, initialize
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from vci.data import MultiDatasetSentences, MultiDatasetSentenceCollator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tests")


def test_data_index():
    with initialize(version_base=None, config_path="../conf"):
        cfg = compose(config_name="defaults")

        t_0 = time.time()
        dataset = MultiDatasetSentences(cfg)
        multi_dataset_sentence_collator = MultiDatasetSentenceCollator(cfg)

        # Make the dataloader outside of the
        dataloader = DataLoader(dataset,
                                batch_size=cfg.model.batch_size,
                                shuffle=True,
                                collate_fn=multi_dataset_sentence_collator,
                                num_workers=8,
                                persistent_workers=True)

        t_1 = time.time()
        log.info(f'Time to load dataset {t_1 - t_0}')

        for i, batch in enumerate(dataloader):
            log.info(f'new_batch_{i}.pickle')
            assert batch[0].shape == (cfg.model.batch_size, cfg.dataset.pad_length)
            assert batch[1].shape == (cfg.model.batch_size, cfg.dataset.pad_length)

            assert batch[2].shape == (cfg.model.batch_size, cfg.model.sample_size)
            assert batch[3].shape == (cfg.model.batch_size, cfg.model.sample_size)

            assert batch[4].shape == (cfg.model.batch_size,)
            assert batch[5].shape == (cfg.model.batch_size,)
            # import pickle
            # with open(f'new_batch_{i}.pickle', 'wb') as handle:
            #     pickle.dump(batch, handle)
            if i > 10:
                break

        log.info(f'Time to iterate thru datset {time.time() - t_1}')
