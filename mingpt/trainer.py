"""
Simple training loop; Boilerplate that could apply to any arbitrary neural network,
so nothing in this file really has anything to do with GPT specifically.
"""

import uuid
import time
import math
import logging
import os

from tqdm import tqdm
import numpy as np

import torch
from torch.utils.data import Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import socket
from dataclasses import dataclass
from typing import Optional
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def get_fq_hostname() -> str:
    return socket.getfqdn(socket.gethostname())


@dataclass
class TrainerConfig:
    max_epochs: int = 10
    batch_size: int = 64
    learning_rate: float = 3e-4
    weight_decay: float = 0.1  # only applied on matmul weights
    # learning rate decay params: linear warmup followed by cosine decay to 10% of original
    lr_decay: bool = False
    warmup_tokens: int = 375e6  # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
    final_tokens: int = 260e9  # (at what point we reach 10% of original LR)
    ckpt_path: Optional[str] = None
    data_loader_workers: int = 0
    enable_profile: bool = False
    log_dir: Optional[str] = None


class Trainer:

    def __init__(self,
                 model: torch.nn.Module,
                 optimizer: optim.Optimizer,
                 train_dataset: Dataset,
                 config: TrainerConfig,
                 device: Optional[int] = None):
        self.model = model
        self.optimizer = optimizer
        self.train_dataset = train_dataset
        self.config = config

        self.device = device
        self.rank = int(os.environ['RANK'])
        self.world_size = int(os.environ['WORLD_SIZE'])
        self.tb_writer = self._get_tb_writer()

    def _get_tb_writer(self) -> Optional[SummaryWriter]:
        if self.config.log_dir:
            return SummaryWriter(log_dir=self.config.log_dir)
        else:
            return None

    def _try_create_profiler(self) -> Optional[torch.profiler.profile]:
        if not self.config.enable_profile:
            return None
        return torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=5),
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.config.log_dir),
        )

    def run_batch(self, epoch, it, x, y):
        with torch.set_grad_enabled(True):
            _, loss = self.model(x, y)

        self.model.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        if self.tb_writer:
            self.tb_writer.add_scalar("batch_loss", loss.item(), it)
        print(
            f"{self.rank}: epoch {epoch + 1} iter {it}: train loss {loss.item():.5f}")

    def run_epoch(self, epoch):
        self.model.train(True)
        data = self.train_dataset
        train_sampler = DistributedSampler(data, rank=self.rank, num_replicas=self.world_size, shuffle=True)
        loader = DataLoader(data,
                            pin_memory=True,
                            batch_size=self.config.batch_size,
                            num_workers=self.config.data_loader_workers,
                            sampler=train_sampler
                            )

        prof = self._try_create_profiler()
        try:
            for it, (x, y) in enumerate(loader):
                x = x.to(self.device)
                y = y.to(self.device)
                self.run_batch(epoch, it, x, y)
                if prof:
                    prof.step()

        finally:
            if prof:
                prof.stop()
            if self.tb_writer:
                self.tb_writer.flush()

    def fit(self):
        for epoch in range(self.config.max_epochs):
            self.run_epoch(epoch)
