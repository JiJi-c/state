import os
import torch
import lightning as L

from torch import nn
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from vci.model import LitUCEModel
from vci.data import H5adDatasetSentences, VCIDatasetSentenceCollator
from vci.train.callbacks import LogLR, PerformanceMonitorCallback
from vci.utils import get_latest_checkpoint, parse_chk_info


def get_ESM2_embeddings(cfg):
    # Load in ESM2 embeddings and special tokens
    all_pe = torch.load(cfg.embeddings.esm2.checkpoint)
    if all_pe.shape[0] == 143574:
        torch.manual_seed(23)
        #MASK_TENSOR = torch.normal(mean=0, std=1, size=(1, args.token_dim))
        CHROM_TENSORS = torch.normal(mean=0, std=1, size=(1895, cfg.tokenizer.token_dim))
        # 1894 is the total number of chromosome choices, it is hardcoded for now
        all_pe = torch.vstack((all_pe, CHROM_TENSORS)) # Add the chrom tensors to the end
        all_pe.requires_grad = False

    #print("Loaded PE", all_pe.shape)
    # randomize it!
    all_pe = torch.randn_like(all_pe) # random init :)
    return all_pe


def main(cfg):
    TOTAL_N_CELL = cfg.dataset.num_cells
    EPOCH_LENGTH = int(TOTAL_N_CELL // cfg.model.batch_size // 24)
    warmup_steps = EPOCH_LENGTH * 6 # ? not sure why this needs to be included but seems empirical?? no clue why this is 6

    dataset_sentence_collator = VCIDatasetSentenceCollator(cfg)

    # Training dataloader
    train_dataset = H5adDatasetSentences(cfg)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=cfg.model.batch_size,
                                  shuffle=False,
                                  collate_fn=dataset_sentence_collator,
                                  num_workers=3,
                                  persistent_workers=True)

    val_dataset = H5adDatasetSentences(cfg, test=True)
    val_dataloader = DataLoader(val_dataset,
                        batch_size=cfg.model.batch_size,
                        shuffle=False,
                        collate_fn=dataset_sentence_collator,
                        num_workers=3,
                        persistent_workers=True)

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
    all_pe.requires_grad= False
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

    wandb_logger = WandbLogger(project=cfg.wandb.project, name=cfg.experiment.name)
    wandb_logger.watch(model, log_freq=1000)

    val_interval = int(cfg.experiment.val_check_interval * cfg.experiment.num_gpus_per_node * cfg.experiment.num_nodes)
    trainer = L.Trainer(max_epochs=cfg.experiment.num_epochs,
                        callbacks=[checkpoint_callback,
                                   LogLR(100),
                                   RichProgressBar(),
                                   PerformanceMonitorCallback()],
                        devices=cfg.experiment.num_gpus_per_node,
                        num_nodes=cfg.experiment.num_nodes,
                        # Accumulation
                        gradient_clip_val=cfg.optimizer.max_grad_norm,
                        accumulate_grad_batches=cfg.optimizer.gradient_accumulation_steps,
                        # precision="bf16-mixed",
                        strategy=DDPStrategy(process_group_backend="nccl"),
                        val_check_interval=val_interval,
                        # Logging
                        logger=wandb_logger,
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
