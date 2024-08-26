import os
import glob
import torch
import lightning as L

from torch import nn
from torch.utils.data import DataLoader

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.strategies import DDPStrategy

from vci.model import LitUCEModel
from vci.data import MultiDatasetSentences, MultiDatasetSentenceCollator
from vci.train.callbacks import LogLR


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

    # Setup Data
    dataset = MultiDatasetSentences(cfg)
    multi_dataset_sentence_collator = MultiDatasetSentenceCollator(cfg)

    # Make the dataloader outside of the
    dataloader = DataLoader(dataset,
                            batch_size=cfg.model.batch_size,
                            shuffle=True,
                            collate_fn=multi_dataset_sentence_collator,
                            num_workers=8,
                            persistent_workers=True)

    model_run_name = "exp_{0}_layers_{1}_dmodel_{2}_samples_{3}_max_lr_{4}_op_dim_{5}".format(
        cfg.experiment.name,
        cfg.model.nlayers,
        cfg.model.emsize,
        cfg.model.sample_size,
        cfg.optimizer.max_lr,
        cfg.model.output_dim)

    chks = glob.glob(os.path.join(cfg.experiment.path, f'{model_run_name}*'))
    if chks:
        model = LitUCEModel.load_from_checkpoint(sorted(chks)[-1])
        print(f'******** Loding chkpoint {model_run_name} {sorted(chks)[-1]}...')
    else:
        print(f'******** Initialized fresh {model_run_name}...')
        model = LitUCEModel(token_dim=cfg.tokenizer.token_dim,
                            d_model=cfg.model.emsize,
                            nhead=cfg.model.nhead,
                            d_hid=cfg.model.d_hid,
                            nlayers=cfg.model.nlayers,
                            output_dim=cfg.model.output_dim,
                            dropout=cfg.model.dropout,
                            warmup_steps=warmup_steps,
                            gradient_accumulation_steps=cfg.optimizer.gradient_accumulation_steps,
                            compiled=False,
                            num_datasets=len(dataset.datasets),
                            max_lr=cfg.optimizer.max_lr,
                            emb_cnt=cfg.embeddings.esm2.cnt,
                            emb_size=cfg.embeddings.esm2.size).cuda()
        all_pe = get_ESM2_embeddings(cfg)
        all_pe.requires_grad= False
        model.pe_embedding = nn.Embedding.from_pretrained(all_pe)

    model = model.train()
    if cfg.experiment.compiled:
        model = torch.compile(model, dynamic=False)

    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.experiment.path,
        filename=f"{model_run_name}"+"-{epoch}-{step}",
        # every_n_train_steps=cfg.experiment.checkpoint.every_n_train_steps,
        save_top_k=cfg.experiment.checkpoint.save_top_k,
        monitor='train_loss',
    )

    wandb_logger = WandbLogger(project=cfg.model.name, name=cfg.experiment.name)
    wandb_logger.watch(model, log_freq=1000)

    trainer = L.Trainer(max_epochs=cfg.experiment.num_epochs,
                        callbacks=[checkpoint_callback,
                                   LogLR(100),
                                   RichProgressBar()],
                        devices=cfg.experiment.num_gpus_per_node,
                        num_nodes=cfg.experiment.num_nodes,
                        # Accumulation
                        gradient_clip_val=cfg.optimizer.max_grad_norm,
                        accumulate_grad_batches=cfg.optimizer.gradient_accumulation_steps,
                        precision="bf16-mixed",
                        strategy=DDPStrategy(process_group_backend="nccl"),
                        # Logging
                        logger=wandb_logger,
                        #profiler=PyTorchProfiler(),
                        fast_dev_run=False,
                       )
    trainer.fit(model=model, train_dataloaders=dataloader)

    trainer.save_checkpoint(os.path.join(cfg.experiment.path,
                            f"{model_run_name}_final.pt"))
