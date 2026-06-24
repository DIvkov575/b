import torch
from torch.utils.data import Dataset
from pathlib import Path


VOCAB_SIZE = 28  # 0=mask, 1-26=a-z, 27=space
MASK_TOKEN = 0   # reserve 0 for mask; chars are 1-27


def char_to_idx(c: str) -> int:
    if c == ' ':
        return 27
    return ord(c) - ord('a') + 1


def text8_tokenize(text: str) -> torch.Tensor:
    return torch.tensor([char_to_idx(c) for c in text], dtype=torch.long)


class Text8Dataset(Dataset):
    """Text8 dataset: 100M characters of Wikipedia, alphabet + space only."""

    def __init__(self, split: str = "train", seq_len: int = 256,
                 max_samples: int = None, data_dir: str = "data"):
        path = Path(data_dir) / "text8"
        if not path.exists():
            self._download(data_dir)
        with open(path) as f:
            raw = f.read()
        if split == "train":
            raw = raw[:90_000_000]
        elif split == "val":
            raw = raw[90_000_000:95_000_000]
        else:
            raw = raw[95_000_000:100_000_000]
        n_seqs = len(raw) // seq_len
        if max_samples:
            n_seqs = min(n_seqs, max_samples)
        self.seqs = torch.zeros(n_seqs, seq_len, dtype=torch.long)
        for i in range(n_seqs):
            chunk = raw[i * seq_len:(i + 1) * seq_len]
            self.seqs[i] = text8_tokenize(chunk)
        self.seq_len = seq_len

    def _download(self, data_dir: str):
        import urllib.request, zipfile, os
        Path(data_dir).mkdir(parents=True, exist_ok=True)
        url = "http://mattmahoney.net/dc/text8.zip"
        zip_path = Path(data_dir) / "text8.zip"
        urllib.request.urlretrieve(url, zip_path)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(data_dir)
        os.remove(zip_path)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def mask_sequence(x: torch.Tensor, t, mask_token: int = MASK_TOKEN) -> torch.Tensor:
    """Apply absorbing-state (masking) noise at time t.
    Each position is independently masked with probability 1-e^{-t}.
    t: float or (B,) tensor. x: (L,) or (B, L).
    """
    if not isinstance(t, torch.Tensor):
        t = torch.tensor([t], dtype=torch.float, device=x.device)
    if t.dim() == 0:
        t = t.unsqueeze(0)
    mask_prob = 1.0 - torch.exp(-t)
    if x.dim() == 2:
        mask_prob = mask_prob.unsqueeze(-1)  # (B, 1)
    mask = torch.rand_like(x.float()) < mask_prob
    return torch.where(mask, torch.full_like(x, mask_token), x)
