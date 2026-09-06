import torch
import torch.nn as nn
import torch.nn.functional as F

class edl_module(nn.Module):
    """
    Evidential Deep Learning module for uncertainty estimation.

    Uses a self-correlation matrix to model Dirichlet evidence, then computes:
      - NLL loss: pushes diagonal evidence higher (each sample should match itself)
      - KL divergence: regularizes against a uniform Dirichlet prior
      - Vacuity: per-sample uncertainty derived from total evidence strength
    """

    def __init__(self, type='relu', kl_weight=0.01):
        super(edl_module, self).__init__()
        self.type = type
        self.kl_weight = kl_weight  # Weight for KL regularization term

    def loglikelihood_loss(self, target, alpha):
        """
        Dirichlet NLL loss: E_ll = ||y - mean||^2 + variance
            mean        = alpha / S
            variance    = sum_i alpha_i (S - alpha_i) / [S^2 (S + 1)]

        Args:
            target: (N, K) one-hot labels
            alpha:  (N, K) Dirichlet parameters (alpha >= 1)
        Returns:
            (N, 1) per-sample loss
        """
        S = torch.sum(alpha, dim=1, keepdim=True)
        # ||y - alpha/S||^2: prediction error
        loglikelihood_err = torch.sum((target - (alpha / S)) ** 2, dim=1, keepdim=True)
        # Dirichlet variance: inherent data ambiguity
        loglikelihood_var = torch.sum(
            alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
        )
        return loglikelihood_err + loglikelihood_var

    def dirichlet_nll(self, target, alpha):
        """Alias for loglikelihood_loss."""
        return self.loglikelihood_loss(target, alpha)

    def kl_divergence(self, alpha):
        """
        KL[Dir(alpha) || Dir(1,...,1)]:
            = ln Gamma(S) - sum ln Gamma(alpha_i)
              - ln Gamma(K) + ln Gamma(1)*K
              + sum (alpha_i - 1)(psi(alpha_i) - psi(S))

        Prevents evidence from growing without bound when data is limited.

        Args:
            alpha: (N, K) Dirichlet parameters
        Returns:
            (N, 1) per-sample KL divergence
        """
        K = alpha.shape[1]
        beta = torch.ones_like(alpha)  # uniform prior
        S_alpha = torch.sum(alpha, dim=1, keepdim=True)
        S_beta = torch.sum(beta, dim=1, keepdim=True)

        kl = (
            torch.lgamma(S_alpha)
            - torch.lgamma(alpha).sum(dim=1, keepdim=True)
            - torch.lgamma(S_beta)
            + torch.lgamma(beta).sum(dim=1, keepdim=True)
            + (alpha - beta) * (torch.digamma(alpha) - torch.digamma(S_alpha))
        ).sum(dim=1, keepdim=True)

        return kl

    def forward(self, matrix):
        """
        Forward pass: compute loss and per-sample uncertainty.

        Args:
            matrix: (N, N) self-similarity matrix (e.g. cosine similarity)
        Returns:
            loss:        scalar, NLL + KL term
            uncertainty: (N, 1) vacuity in [0, 1], 0=certain, 1=uncertain
        """
        N = matrix.shape[0]

        # Target = I: each sample should perfectly match itself
        target = torch.eye(N, dtype=matrix.dtype, device=matrix.device)

        # ReLU clips negative values, +1 shifts to valid Dirichlet params (alpha >= 1)
        evidence = F.relu(matrix)
        alpha = evidence + 1
        S = torch.sum(alpha, dim=1, keepdim=True)

        # Symmetric NLL: forward + transposed, averaged
        nll_fwd = self.dirichlet_nll(target, alpha)
        nll_twd = self.dirichlet_nll(target, alpha.T)
        nll_loss = (nll_fwd.mean() + nll_twd.mean()) / 2.0

        # KL regularization
        kl_term = self.kl_weight * self.kl_divergence(alpha).mean()

        loss = nll_loss + kl_term

        # Vacuity = 1 - K/S; larger S (more evidence) -> lower uncertainty
        uncertainty = (1 - N / S).clamp(0, 1)

        return loss, uncertainty

if __name__ == '__main__':
    edl = edl_module(kl_weight=0.01)

    # Simulate 32 samples with pairwise similarities
    matrix = torch.randn(32, 32)
    loss, uncertainty = edl(matrix)

    print("matrix shape:", matrix.shape)
    print("loss:", loss.item())
    # print("uncertainty (vacuity):\n", uncertainty)
