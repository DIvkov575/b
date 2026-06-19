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


def generate_markov_sequences(n: int, K: int, L: int) -> torch.Tensor:
    """Generate sequences from a structured transition matrix.
    Creates a random stochastic matrix with self-transition bias,
    producing sequences with learnable inter-position patterns.
    """
    torch.manual_seed(42)
    # Transition matrix: prefer self-transition and neighbors
    T = torch.zeros(K, K)
    for i in range(K):
        T[i, i] = 3.0
        T[i, (i + 1) % K] = 2.0
        T[i, (i - 1) % K] = 1.5
    T = T + 0.1
    T = T / T.sum(dim=1, keepdim=True)

    # Generate sequences
    seqs = torch.zeros(n, L, dtype=torch.long)
    start_probs = torch.ones(K) / K
    start_probs[0] = 2.0
    start_probs[1] = 1.5
    start_probs = start_probs / start_probs.sum()
    seqs[:, 0] = torch.multinomial(start_probs.expand(n, -1), 1).squeeze(-1)

    for l in range(1, L):
        prev = seqs[:, l - 1]
        trans_probs = T[prev]
        seqs[:, l] = torch.multinomial(trans_probs, 1).squeeze(-1)

    torch.manual_seed(torch.seed())
    return seqs


class MarkovSequenceDataset(Dataset):
    """Structured sequences from a Markov chain with computable properties."""

    def __init__(self, n: int = 10000, K: int = 8, L: int = 32):
        self.seqs = generate_markov_sequences(n, K, L)
        self.labels = {
            "starts_with_0": property_starts_with_0(self.seqs),
            "contains_pattern": property_contains_pattern(self.seqs),
            "no_repeats": property_no_adjacent_repeats(self.seqs),
        }
        self.n_properties = len(self.labels)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx]
        props = {k: v[idx].item() for k, v in self.labels.items()}
        return seq, props
