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
        q = F.softplus(quality_scores).clamp(max=20.0)
        normalized = F.normalize(embeddings, p=2, dim=-1)
        S = normalized @ normalized.T
        L = q.unsqueeze(1) * S * q.unsqueeze(0)
        return L

    def forward(self, embeddings, quality_scores):
        """Greedy MAP DPP via Cholesky-based incremental updates (numerically stable)."""
        n = embeddings.shape[0]
        if n <= self.budget_k:
            return list(range(n))

        L = self._build_L_kernel(embeddings, quality_scores)
        diag_L = torch.diag(L).clone()

        selected: list[int] = []
        # Cholesky rows for incremental Schur complement
        chol_rows = torch.zeros(self.budget_k, n, device=L.device, dtype=L.dtype)
        remaining = torch.ones(n, dtype=torch.bool, device=L.device)

        for t in range(self.budget_k):
            # Conditional gains = diag_L (updated in place)
            gains = diag_L.clone()
            gains[~remaining] = -float("inf")

            if gains.max() <= 0:
                break

            best = int(gains.argmax().item())
            selected.append(best)
            remaining[best] = False

            # Update Cholesky factor
            if t == 0:
                chol_rows[0, best] = torch.sqrt(diag_L[best].clamp(min=1e-10))
            else:
                prev_chol = chol_rows[:t, best]
                chol_rows[t, best] = torch.sqrt((diag_L[best] - prev_chol @ prev_chol).clamp(min=1e-10))

            # Update conditional gains for remaining items
            L_col = L[best, :]
            if t == 0:
                e = L_col / chol_rows[0, best]
            else:
                prev = chol_rows[:t, :]
                solve_rhs = L_col - (prev[:, best] @ prev)
                e = solve_rhs / chol_rows[t, best]

            chol_rows[t, :] = e
            diag_L -= e * e

        return selected

    def soft_select(self, embeddings, quality_scores):
        """Differentiable relaxation: marginal inclusion probabilities via K = L(L+I)^{-1}."""
        n = embeddings.shape[0]
        L = self._build_L_kernel(embeddings, quality_scores)
        eye = torch.eye(n, device=L.device, dtype=L.dtype)
        # Use solve instead of inv for stability: K = L @ (L + I)^{-1}
        # Equivalent: K = I - (L + I)^{-1}, but solve is more stable
        LpI = L + eye
        # Add small regularization to ensure positive-definite
        LpI = LpI + 1e-4 * eye
        K = torch.linalg.solve(LpI.T, L.T).T
        marginals = torch.diagonal(K).clamp(0.0, 1.0)
        return marginals
