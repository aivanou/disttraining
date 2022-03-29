import os
import sys
from typing import Optional, Tuple
from mingpt.model import GPT, GPTConfig, OptimizerConfig, create_optimizer
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.char_dataset import CharDataset
from mingpt.utils import sample
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
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


def get_model_and_optimizer(gpt_config: GPTConfig, opt_config: OptimizerConfig, trainer_config: TrainerConfig) \
        -> Tuple[torch.nn.Module, torch.optim.Optimizer, int]:
    model = GPT(gpt_config)
    optimizer = create_optimizer(model, opt_config)
    start_epoch = 0
    if trainer_config.checkpoint_path and os.path.exists(trainer_config.checkpoint_path):
        checkpoint = torch.load(trainer_config.checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']

    device = get_device()
    device_ids = None
    if device is not None:
        model = model.to(device)
        device_ids = [device]
    model = DistributedDataParallel(
        model,
        device_ids=device_ids,
        find_unused_parameters=True,
    )
    return model, optimizer, start_epoch


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


def get_dataset_path() -> str:
    path = os.path.abspath(__file__)
    dirname = os.path.dirname(path)
    return os.path.join(dirname, "data", "input.txt")

import fsspec
@hydra.main(config_path=".", config_name="trainer_config")
def main(cfg: DictConfig):
    set_env()
    device = get_device()
    print(f"{get_fq_hostname()}:{os.getpid()}:{device} Running charNN")
    if device is not None:
        torch.cuda.set_device(device)
    setup_process_group()

    block_size = 128  # spatial extent of the model for its context
    train_dataset = CharDataset(get_dataset_path(), block_size)

    opt_conf = OptimizerConfig(lr=cfg['opt']['lr'], weight_decay=cfg['opt']['weight_decay'])

    mconf = GPTConfig(vocab_size=train_dataset.vocab_size,
                      block_size=train_dataset.block_size,
                      n_layer=cfg['model']['n_layer'],
                      n_head=cfg['model']['n_head'],
                      n_embd=cfg['model']['n_embd'])

    train_cfg = cfg['trainer']
    tconf = TrainerConfig(max_epochs=train_cfg['max_epochs'],
                          batch_size=train_cfg['batch_size'],
                          data_loader_workers=train_cfg['data_loader_workers'],
                          enable_profile=train_cfg['enable_profile'],
                          log_dir=train_cfg.get('log_dir'),
                          checkpoint_path=train_cfg.get("checkpoint_path"),
                          )
    model, optimizer, start_epoch = get_model_and_optimizer(mconf, opt_conf, tconf)

    if cfg['charnn']['task'] == 'train':
        trainer = Trainer(model, optimizer, train_dataset, tconf, device, start_epoch)
        trainer.fit()
    elif cfg['charnn']['task'] == 'generate':
        generate_seq(cfg, model, train_dataset)
    else:
        raise RuntimeError(f"Unknown task: {cfg['charnn']['task']}")


if __name__ == "__main__":
    main()
