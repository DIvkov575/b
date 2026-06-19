import torch
from torch.utils.data import Dataset


def generate_uniform_sequences(n: int, K: int, L: int) -> torch.Tensor:
    return torch.randint(0, K, (n, L))


def property_starts_with_0(seqs: torch.Tensor) -> torch.Tensor:
    return seqs[:, 0] == 0


def property_contains_pattern(seqs: torch.Tensor, pattern: list = [1, 2, 3]) -> torch.Tensor:
    pat = torch.tensor(pattern, device=seqs.device)
    pat_len = len(pattern)
    n, L = seqs.shape
    results = torch.zeros(n, dtype=torch.bool, device=seqs.device)
    for i in range(L - pat_len + 1):
        match = (seqs[:, i:i + pat_len] == pat.unsqueeze(0)).all(dim=1)
        results |= match
    return results


def property_no_adjacent_repeats(seqs: torch.Tensor) -> torch.Tensor:
    diffs = seqs[:, 1:] != seqs[:, :-1]
    return diffs.all(dim=1)


PROPERTIES = {
    "starts_with_0": property_starts_with_0,
    "contains_pattern": property_contains_pattern,
    "no_repeats": property_no_adjacent_repeats,
}


class SyntheticSequenceDataset(Dataset):
    def __init__(self, n: int = 10000, K: int = 8, L: int = 32):
        self.seqs = generate_uniform_sequences(n, K, L)
        self.labels = {
            "starts_with_0": property_starts_with_0(self.seqs),
            "contains_pattern": property_contains_pattern(self.seqs),
            "no_repeats": property_no_adjacent_repeats(self.seqs),
        }

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        props = {k: v[idx].item() for k, v in self.labels.items()}
        return seq, props
