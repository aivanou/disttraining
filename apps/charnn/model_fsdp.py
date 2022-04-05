#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
The model taken form https://github.com/karpathy/minGPT
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import logging
import os
from functools import partial

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributed.fsdp.wrap import wrap
from torch.distributed.algorithms._checkpoint._checkpoint_wrapper import checkpoint_wrapper
from model import GPTConfig, MultiheadAttentionLayer, EmbeddingStem

logger = logging.getLogger(__name__)


def module_wrapper(module, fsdp=False, activation="noop"):
    if not fsdp:
        return module

    if activation == "noop":
        return wrap(module)
    elif activation == "checkpoint":
        return wrap(checkpoint_wrapper(module))
    elif activation == "offload":
        return wrap(checkpoint_wrapper(module, offload_to_cpu=True))
    else:
        raise ValueError(f"Unrecognized activation mode {activation}")


class ShardedBlock(nn.Module):

    def __init__(
            self,
            config: GPTConfig,
            device=None,
            dtype=torch.float32,
            wrapper=lambda m: m,
    ):
        super().__init__()
        self.ln1 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
        self.ln2 = wrapper(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
        self.attn = MultiheadAttentionLayer(config, device=device, dtype=dtype)
        self.mlp = nn.Sequential(
            wrapper(nn.Linear(config.n_embd, 4 * config.n_embd, device=device, dtype=dtype)),
            nn.GELU(),
            wrapper(nn.Linear(4 * config.n_embd, config.n_embd, device=device, dtype=dtype)),
            nn.Dropout(config.resid_pdrop),
        )

    def reset_parameters(self):
        self.attn.reset_parameters()
        for _, m in self.named_modules():
            if isinstance(m, nn.LayerNorm) or isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ShardedGPT(nn.Module):
    def __init__(self, config: GPTConfig, device="cpu", dtype=torch.float32, activation="noop"):
        super().__init__()

        wrapper = partial(module_wrapper, fsdp=True, activation=activation)

        # input embedding stem
        self.emb_stem = wrap(EmbeddingStem(config, device=device, dtype=dtype))
        # transformer
        self.blocks = nn.Sequential(
            *[wrapper(ShardedBlock(config, device=device, dtype=dtype, wrapper=wrap)) for _ in range(config.n_layer)]
        )
        # decoder head
        self.ln_f = wrap(nn.LayerNorm(config.n_embd, device=device, dtype=dtype))
        self.head = wrap(nn.Linear(config.n_embd, config.vocab_size, bias=False, device=device, dtype=dtype))
        rank = int(os.getenv("RANK", "0"))

        if rank == 0:
            print("GPT Model Number of parameters:", sum(p.numel() for p in self.parameters()))

    def forward(self, idx, targets=None):
        x = self.emb_stem(idx)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss
