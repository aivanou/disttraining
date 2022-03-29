import os
import tempfile
import fsspec

import torch
from torch.utils.data import Dataset


class CharDataset(Dataset):

    def __init__(self, data_path: str, block_size):
        data = self._read_data(data_path)
        chars = sorted(list(set(data)))
        data_size, vocab_size = len(data), len(chars)
        print('Data has %d characters, %d unique.' % (data_size, vocab_size))

        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.data = data

    def _read_data(self, data_path: str) -> str:
        with tempfile.TemporaryDirectory() as tmpdir:
            fs, _, rpaths = fsspec.get_fs_token_paths(data_path)
            local_file = os.path.join(tmpdir, "input.txt")

            fs.get(rpaths[0], local_file)
            data = open(local_file, 'r').read()
            return data

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
