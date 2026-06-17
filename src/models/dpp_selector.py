import torch
import torch.nn as nn
import torch.nn.functional as F


class DPPSelector(nn.Module):
    def __init__(self, embed_dim, budget_k, temperature=1.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.budget_k = budget_k
        self.temperature = temperature

    def _build_L_kernel(self, embeddings, quality_scores):
        q = F.softplus(quality_scores)
        normalized = F.normalize(embeddings, p=2, dim=-1)
        S = normalized @ normalized.T
        L = q.unsqueeze(1) * S * q.unsqueeze(0)
        return L

    def forward(self, embeddings, quality_scores):
        n = embeddings.shape[0]
        if n <= self.budget_k:
            return list(range(n))

        L = self._build_L_kernel(embeddings, quality_scores)
        device = L.device
        eye = torch.eye(n, device=device, dtype=L.dtype) * 1e-6
        L_reg = L + eye

        selected: list[int] = []
        remaining = set(range(n))

        first = int(torch.argmax(torch.diag(L)).item())
        selected.append(first)
        remaining.remove(first)

        while len(selected) < self.budget_k and remaining:
            sel_idx = torch.tensor(selected, device=device, dtype=torch.long)
            rem_idx = torch.tensor(sorted(remaining), device=device, dtype=torch.long)

            L_sel = L_reg.index_select(0, sel_idx).index_select(1, sel_idx)
            L_cross = L_reg.index_select(0, rem_idx).index_select(1, sel_idx)
            diag_rem = torch.diag(L_reg).index_select(0, rem_idx)

            L_sel_inv = torch.linalg.inv(L_sel)
            gains = diag_rem - torch.sum((L_cross @ L_sel_inv) * L_cross, dim=1)

            best_local = int(torch.argmax(gains).item())
            best_global = int(rem_idx[best_local].item())
            selected.append(best_global)
            remaining.remove(best_global)

        return selected

    def soft_select(self, embeddings, quality_scores):
        n = embeddings.shape[0]
        L = self._build_L_kernel(embeddings, quality_scores)
        eye = torch.eye(n, device=L.device, dtype=L.dtype)
        K = L @ torch.linalg.inv(L + eye + 1e-6 * eye)
        marginals = torch.diagonal(K).clamp(0.0, 1.0)
        return torch.sigmoid((marginals - 0.5) / self.temperature)
