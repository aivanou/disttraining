#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional, Tuple
from model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from model_fsdp import ShardedGPT
from trainer import Trainer, TrainerConfig, Checkpoint, load_checkpoint
from char_dataset import CharDataset
from utils import sample
from torch.nn.parallel import DistributedDataParallel
from torch.distributed.fsdp.wrap import enable_wrap, wrap
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, CPUOffload
import torch.distributed as dist
from torch.utils.data import random_split
import uuid

import torch
import socket
from omegaconf import DictConfig

import hydra


def get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())


def set_env():
    os.environ['RANK'] = os.environ.get("RANK", "0")
    os.environ['WORLD_SIZE'] = os.environ.get("WORLD_SIZE", "1")
    os.environ['MASTER_PORT'] = os.environ.get("MASTER_PORT", "29830")
    os.environ['MASTER_ADDR'] = os.environ.get("MASTER_ADDR", 'localhost')
    os.environ['LOCAL_RANK'] = os.environ.get("LOCAL_RANK", "0")
    os.environ['TORCHELASTIC_RUN_ID'] = os.environ.get("TORCHELASTIC_RUN_ID", str(uuid.uuid4()).split('-')[0])


def get_job_name():
    uid = os.environ['TORCHELASTIC_RUN_ID']
    return f"test-job-{uid}"


def get_device() -> Optional[int]:
    if not torch.cuda.is_available():
        return None
    return int(os.environ['LOCAL_RANK'])


def get_ddp_model_and_optimizer(gpt_config: GPTConfig, opt_config: OptimizerConfig, checkpoint: Optional[Checkpoint]) \
        -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    # Create new GPT Model on CPU
    model = GPT(gpt_config)
    # Load GPT model from checkpoint if present
    if checkpoint:
        model.load_state_dict(checkpoint.model_state)
    device = get_device()
    device_ids = None
    if device is not None:
        model = model.to(device)
        device_ids = [device]
    optimizer = create_optimizer(model, opt_config)
    if checkpoint:
        optimizer.load_state_dict(checkpoint.optimizer_state)
    model = DistributedDataParallel(
        model,
        device_ids=device_ids,
    )
    return model, optimizer


def get_fsdp_model_and_optimizer(gpt_config: GPTConfig, opt_config: OptimizerConfig, checkpoint: Optional[Checkpoint]) \
        -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    device = get_device()
    with enable_wrap(wrapper_cls=FSDP, cpu_offload=CPUOffload(offload_params=True)):
        model = wrap(
            ShardedGPT(gpt_config, device=device, dtype=torch.float32, activation="offload"))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt_config.lr)
    return model, optimizer


def get_model_and_optimizer(type: str, gpt_config: GPTConfig, opt_config: OptimizerConfig,
                            checkpoint: Optional[Checkpoint]) \
        -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    if type == "ddp":
        return get_ddp_model_and_optimizer(gpt_config, opt_config, checkpoint)
    elif type == "fsdp":
        return get_fsdp_model_and_optimizer(gpt_config, opt_config, checkpoint)
    else:
        raise RuntimeError(f"Unknown type: {type}. Allowed values: [ddp, fsdp]")


def setup_process_group() -> None:
    device = get_device()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if device is not None:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


def generate_seq(cfg: DictConfig, model: torch.nn.Module, dataset: CharDataset) -> None:
    if dist.get_rank() == 0:
        device = get_device()
        context = cfg['charnn']['phrase']
        x = torch.tensor([dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(device)
        y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
        completion = ''.join([dataset.itos[int(i)] for i in y])
        print(completion)


def get_dataset_path(cfg: DictConfig) -> str:
    default_path = './data/input.txt'
    if cfg['dataset']['path'] == default_path:
        path = os.path.abspath(__file__)
        dirname = os.path.dirname(path)
        return os.path.join(dirname, "data", "input.txt")
    else:
        return cfg['dataset']['path']


@hydra.main(config_path=".", config_name="trainer_config")
def main(cfg: DictConfig):
    set_env()
    device = get_device()
    data_path = get_dataset_path(cfg)
    job_name = get_job_name()
    print(f"{get_fq_hostname()}:{os.getpid()}:{device} Running charNN {job_name}, data_path: {data_path}")
    if device is not None:
        torch.cuda.set_device(device)
    setup_process_group()

    block_size = 128  # spatial extent of the model for its context
    dataset = CharDataset(data_path, block_size)

    datalen = len(dataset)
    train_len = int(datalen * 0.9)

    train_dataset, test_dataset = random_split(dataset, [train_len, datalen - train_len])

    opt_conf = OptimizerConfig(lr=cfg['opt']['lr'], weight_decay=cfg['opt']['weight_decay'])

    mconf = GPTConfig(vocab_size=dataset.vocab_size,
                      block_size=dataset.block_size,
                      n_layer=cfg['model']['n_layer'],
                      n_head=cfg['model']['n_head'],
                      n_embd=cfg['model']['n_embd'])

    train_cfg = cfg['trainer']
    tconf = TrainerConfig(job_name=job_name,
                          max_epochs=train_cfg['max_epochs'],
                          batch_size=train_cfg['batch_size'],
                          data_loader_workers=train_cfg['data_loader_workers'],
                          enable_profile=train_cfg['enable_profile'],
                          log_dir=train_cfg.get('log_dir'),
                          checkpoint_path=train_cfg.get("checkpoint_path"), )
    checkpoint = load_checkpoint(tconf.checkpoint_path)
    model, optimizer = get_model_and_optimizer(cfg['charnn']['dist'], mconf, opt_conf, checkpoint)

    if cfg['charnn']['task'] == 'train':
        trainer = Trainer(model, optimizer, train_dataset, test_dataset, tconf, device,
                          checkpoint.finished_epoch + 1 if checkpoint else 0)
        trainer.fit()
    elif cfg['charnn']['task'] == 'generate':
        generate_seq(cfg, model, train_dataset)
    else:
        raise RuntimeError(f"Unknown task: {cfg['charnn']['task']}")


if __name__ == "__main__":
    main()
