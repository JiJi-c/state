from data.data_modules import (
    MultiDatasetPerturbationDataModule,
)

DATA_MODULE_DICT = dict(
    MultiDatasetPerturbationDataModule=MultiDatasetPerturbationDataModule,
)


### load data / lightning modules
def get_datamodule(name, kwargs, batch_size=None, cell_sentence_len=1):
    if name not in DATA_MODULE_DICT:
        raise ValueError(f"Unknown data module '{name}'")

    if batch_size is not None:
        kwargs["batch_size"] = batch_size
        kwargs["cell_sentence_len"] = cell_sentence_len

    return DATA_MODULE_DICT[name](**kwargs)
