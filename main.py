import os
from typing import Optional, Tuple
from mingpt.model import GPT, GPTConfig
from mingpt.trainer import Trainer, TrainerConfig
from mingpt.utils import sample
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import uuid

from torch.utils.data import Dataset
import torch
import socket
from omegaconf import DictConfig

import hydra


def get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())


class CharDataset(Dataset):

    def __init__(self, data, block_size):
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # grab a chunk of (block_size + 1) characters from the data
        chunk = self.data[idx:idx + self.block_size + 1]
        # encode every character to an integer
        dix = [self.stoi[s] for s in chunk]
        x = torch.tensor(dix[:-1], dtype=torch.long)
        y = torch.tensor(dix[1:], dtype=torch.long)
        return x, y


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


def get_model_and_optimizer(gpt_config: GPTConfig, trainer_config: TrainerConfig) -> Tuple[
    torch.nn.Module, torch.optim.Optimizer]:
    model = GPT(gpt_config)
    optimizer = model.configure_optimizers(trainer_config)
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
    return model, optimizer


def setup_process_group() -> None:
    device = get_device()
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    if device is not None:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)


@hydra.main(config_path=".", config_name="trainer_config")
def main(cfg: DictConfig):
    set_env()
    os.getcwd()
    text = open('./data/input.txt', 'r').read()
    device = get_device()
    print(f"{get_fq_hostname()}:{os.getpid()}:{device} Running charNN")
    if device is not None:
        torch.cuda.set_device(device)
    setup_process_group()

    block_size = 128  # spatial extent of the model for its context
    train_dataset = CharDataset(text, block_size)  # one line of poem is roughly 50 characters

    mconf = GPTConfig(train_dataset.vocab_size, train_dataset.block_size,
                      n_layer=cfg['model']['n_layer'],
                      n_head=cfg['model']['n_head'],
                      n_embd=cfg['model']['n_embd'])

    tconf = TrainerConfig(max_epochs=cfg['trainer']['max_epochs'],
                          batch_size=cfg['trainer']['batch_size'],
                          learning_rate=cfg['trainer']['lr'],
                          lr_decay=cfg['trainer']['lr_decay'],
                          warmup_tokens=512 * 20,
                          final_tokens=2 * len(train_dataset) * block_size,
                          data_loader_workers=cfg['trainer']['data_loader_workers'],
                          enable_profile=cfg['trainer']['enable_profile'],
                          log_dir=cfg['trainer'].get('log_dir'),
                          )
    model, optimizer = get_model_and_optimizer(mconf, tconf)

    trainer = Trainer(model, optimizer, train_dataset, tconf, device)
    trainer.fit()
    if dist.get_rank() == 0:
        context = "Hello my"
        x = torch.tensor([train_dataset.stoi[s] for s in context], dtype=torch.long)[None, ...].to(device)
        y = sample(model, x, 2000, temperature=1.0, sample=True, top_k=10)[0]
        completion = ''.join([train_dataset.itos[int(i)] for i in y])
        print(completion)


if __name__ == "__main__":
    main()
