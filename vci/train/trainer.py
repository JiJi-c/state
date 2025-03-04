import os
import torch
import lightning as L

from torch import nn
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from vci.nn.model import LitUCEModel
from vci.data import H5adDatasetSentences, VCIDatasetSentenceCollator, FilteredGenesCounts
from vci.train.callbacks import LogLR, ProfilerCallback
from vci.utils import get_latest_checkpoint


def get_ESM2_embeddings(cfg):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(cfg.embeddings.esm2.embedding_file)
    all_pe = torch.vstack(list(all_pe.values()))
    all_pe.requires_grad = False
    return all_pe

def main(cfg):
    TOTAL_N_CELL = cfg.dataset.num_cells
    EPOCH_LENGTH = int(TOTAL_N_CELL // cfg.model.batch_size // 24)
    # ? not sure why this needs to be included but seems empirical?? no clue why this is 6
    warmup_steps = EPOCH_LENGTH * 6

    dataset_sentence_collator = VCIDatasetSentenceCollator(cfg)

    generator = torch.Generator()
    generator.manual_seed(cfg.dataset.seed)

    DatasetClass = FilteredGenesCounts if cfg.dataset.filter else H5adDatasetSentences

    # Training dataloader
    train_dataset = DatasetClass(cfg)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.model.batch_size,
                                  shuffle=False,
                                  collate_fn=dataset_sentence_collator,
                                  num_workers=4,
                                  persistent_workers=True,
                                  generator=generator)

    val_dataset = DatasetClass(cfg, test=True)
    val_dataloader = DataLoader(val_dataset,
                                batch_size=cfg.model.batch_size,
                                shuffle=False,
                                collate_fn=dataset_sentence_collator,
                                num_workers=4,
                                persistent_workers=True,
                                generator=generator)

    model = LitUCEModel(token_dim=cfg.tokenizer.token_dim,
                        d_model=cfg.model.emsize,
                        nhead=cfg.model.nhead,
                        d_hid=cfg.model.d_hid,
                        nlayers=cfg.model.nlayers,
                        output_dim=cfg.model.output_dim,
                        dropout=cfg.model.dropout,
                        warmup_steps=warmup_steps,
                        compiled=False,
                        max_lr=cfg.optimizer.max_lr,
                        emb_cnt=cfg.embeddings.esm2.cnt,
                        emb_size=cfg.embeddings.esm2.size,
                        cfg=cfg).cuda()
    all_pe = get_ESM2_embeddings(cfg)
    all_pe.requires_grad = False
    model.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    model = model.train()
    if cfg.experiment.compiled:
        model = torch.compile(model, dynamic=False)

    run_name, chk = get_latest_checkpoint(cfg)
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=cfg.experiment.checkpoint.every_n_train_steps,
        dirpath=os.path.join(cfg.experiment.checkpoint.path, cfg.experiment.name),
        filename=f"{run_name}"+"-{epoch}-{step}",
        save_last=True,
        save_top_k=cfg.experiment.checkpoint.save_top_k,
        monitor=cfg.experiment.checkpoint.monitor,
    )

    if cfg.wandb.enable:
        exp_logger = WandbLogger(project=cfg.wandb.project, name=cfg.experiment.name)
        exp_logger.watch(model, log_freq=1000)
    else:
        exp_logger = None

    callbacks = [checkpoint_callback,
                 LogLR(100),
                 RichProgressBar()]

    max_steps = -1
    if cfg.experiment.profile.enable_profiler:
        callbacks.append(ProfilerCallback(cfg=cfg))
        max_steps = cfg.experiment.profile.max_steps

    val_interval = int(cfg.experiment.val_check_interval * cfg.experiment.num_gpus_per_node * cfg.experiment.num_nodes)
    trainer = L.Trainer(max_epochs=cfg.experiment.num_epochs,
                        max_steps=max_steps,
                        callbacks=callbacks,
                        devices=cfg.experiment.num_gpus_per_node,
                        num_nodes=cfg.experiment.num_nodes,
                        # Accumulation
                        gradient_clip_val=cfg.optimizer.max_grad_norm,
                        accumulate_grad_batches=cfg.optimizer.gradient_accumulation_steps,
                        precision="bf16-mixed",
                        strategy=DDPStrategy(process_group_backend="nccl"),
                        val_check_interval=val_interval,
                        # Logging
                        logger=exp_logger,
                        fast_dev_run=False,
                        limit_val_batches=cfg.experiment.limit_val_batches,
                        )

    if chk:
        print(f'******** Loading chkpoint {run_name} {chk}...')
    else:
        print(f'******** Initialized fresh {run_name}...')

    trainer.fit(model=model,
                train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader,
                ckpt_path=chk)

    trainer.save_checkpoint(os.path.join(cfg.experiment.checkpoint.path,
                            f"{run_name}_final.pt"))
