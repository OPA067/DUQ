import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention block with pre-norm and learned attention pooling.

    Architecture:
        1. **Pre-norm Transformer block**:
           LayerNorm -> MultiheadAttention (self-attention) -> residual add ->
           LayerNorm -> MLP (expand 4x, GELU, project back) -> residual add.
        2. **Learned attention pooling**:
           A lightweight MLP produces per-token attention scores, masked softmax
           over the sequence, then einsum-weighted sum collapses the sequence
           dimension into a single vector per sample.

    The combination of self-attention followed by learned pooling allows the
    module to first enrich token representations via contextual interactions,
    then distill the sequence into a compact embedding suitable for downstream
    similarity computation.
    """

    def __init__(self, num_heads, embed_dim, attn_mask=None):
        """
        Args:
            num_heads (int): number of parallel attention heads. Must divide
                ``embed_dim`` evenly (PyTorch default requirement).
            embed_dim (int): feature dimension of each token (d_model).
            attn_mask (Tensor or callable, optional): causal or padding mask passed
                to ``nn.MultiheadAttention``. If callable, it is invoked with
                ``batch_size`` to produce a dynamic mask per forward call.
        """
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # ------------------------------------------------------------------
        # Pre-norm Transformer block
        # ------------------------------------------------------------------
        # LayerNorm before attention and before MLP (pre-norm variant) for
        # training stability, especially when the module is inserted on top
        # of a frozen CLIP backbone.
        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln_2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.attn_mask = attn_mask

        # ------------------------------------------------------------------
        # Learned attention pooling
        # ------------------------------------------------------------------
        # Maps each token to a scalar importance score. The scores go through
        # softmax over the valid sequence positions, yielding a probability
        # distribution used to compute a weighted average of the token features.
        self.attn_pool = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, 1),
        )

    def forward(self, x, mask=None):
        """Forward pass: self-attention enrichment + learned pooling.

        Args:
            x (Tensor): input token features, shape ``[B, N, D]``.
                B = batch size, N = sequence length, D = embed_dim.
            mask (BoolTensor or ByteTensor, optional): validity mask of shape
                ``[B, N]``. Non-zero / True positions are valid; padding positions
                receive ``-inf`` attention weight and are excluded from pooling.

        Returns:
            Tensor: pooled features, shape ``[B, D]`` (if N > 1) or ``[B, D]``
            after squeezing the singleton sequence dimension (if N == 1).
        """
        # ------------------------------------------------------------------
        # Transformer block (pre-norm)
        # ------------------------------------------------------------------
        # Handle dynamic causal mask (e.g. for autoregressive models)
        attn_mask_ = self.attn_mask
        if attn_mask_ is not None and callable(attn_mask_):
            attn_mask_ = attn_mask_(x.size(0))
        attn_mask_ = attn_mask_.to(dtype=x.dtype, device=x.device) if attn_mask_ is not None else None

        # Self-attention with residual. normed is the LayerNorm-ed input.
        # The attention output is added back to the original x (residual).
        normed = self.ln_1(x)
        h = x + self.attn(normed, normed, normed, need_weights=False, attn_mask=attn_mask_)[0]
        # MLP with residual. Pre-norm variant: LayerNorm before MLP.
        x = h + self.mlp(self.ln_2(h))

        # ------------------------------------------------------------------
        # Attention pooling: collapse sequence dimension -> single vector
        # ------------------------------------------------------------------
        # Compute a scalar score per token, mask pad positions to -inf,
        # then softmax-normalize across the valid tokens.
        attn = self.attn_pool(x).squeeze(-1)       # [B, N]
        if mask is not None:
            # Inverted mask: False / 0 -> pad -> -inf
            attn.masked_fill_(~mask.bool(), -np.inf)
        attn = torch.softmax(attn, dim=-1)         # [B, N]

        # Weighted sum across the sequence: sum_n (x[b,n,d] * attn[b,n])
        output = torch.einsum("bnd,bn->bd", [x, attn])   # [B, D]

        # If the sequence has length 1 (e.g. single global token), squeeze out
        # the redundant dimension to return a strict 2-D tensor [B, D].
        return output.squeeze(1) if output.shape[1] == 1 else output


class ProbEmbedModule(nn.Module):
    """Probabilistic Embedding Module: deterministic features -> Gaussian parameters.

    Maps a sequence of token features into two outputs:
      - **mu**    : the mean of a diagonal Gaussian distribution (L2-normalized).
      - **sigma** : the log-standard-deviation of the diagonal Gaussian.

    Architecture:
        Two parallel branches, each consisting of:
        MultiHeadSelfAttention (contextual enrichment + pooling)
        -> MLP (Linear-ReLU-Linear)
        -> LayerNorm
        The mu branch additionally applies L2 normalization to constrain the mean
        vector on the unit hypersphere, which stabilizes contrastive learning.

    The resulting (mu, sigma) pair is consumed by ``sample_gaussian`` to draw
    Monte-Carlo samples for stochastic similarity computation in the retrieval model.
    """

    def __init__(self, num_heads=8, embed_dim=512):
        """
        Args:
            num_heads (int): number of attention heads in both mu and sigma branches.
            embed_dim (int): feature dimension. **Caveat**: this is hardcoded to 512
                by default. If the upstream CLIP backbone uses a different dim
                (e.g. 768 for ViT-L), this must be overridden via the constructor
                argument from the caller.
        """
        super(ProbEmbedModule, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim

        # ------------------------------------------------------------------
        # Mu branch: mean of the diagonal Gaussian
        # ------------------------------------------------------------------
        # Produces the central tendency vector of the probabilistic embedding.
        self.attention_mu = MultiHeadSelfAttention(self.num_heads, self.embed_dim)
        self.mlp_mu = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # ------------------------------------------------------------------
        # Sigma branch: log-standard-deviation of the diagonal Gaussian
        # ------------------------------------------------------------------
        # Produces the uncertainty / spread of the embedding.
        self.attention_sigma = MultiHeadSelfAttention(self.num_heads, self.embed_dim)
        self.mlp_sigma = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )

        # Separate LayerNorms for mu and sigma to avoid cross-branch statistics
        # contamination during training.
        self.ln_1 = nn.LayerNorm(embed_dim)   # post-processing for mu
        self.ln_2 = nn.LayerNorm(embed_dim)   # post-processing for sigma

    def l2_normalize(self, tensor, axis=-1):
        """L2-normalize along the specified axis.

        Args:
            tensor (Tensor): input tensor of arbitrary shape.
            axis (int): dimension along which to compute the L2 norm.

        Returns:
            Tensor: same shape as input, with unit L2 norm along ``axis``.
        """
        return F.normalize(tensor, p=2, dim=axis)

    def sample_gaussian(self, mu, sigma, nums=10):
        """Reparameterization trick: sample from N(mu, exp(sigma)^2).

        Standard Gaussian reparameterization:
            z = mu + exp(sigma) * epsilon,
            where epsilon ~ N(0, I)

        We draw ``nums`` independent samples per input vector, producing a
        stochastic set of representations for Monte-Carlo estimation of the
        expected similarity in the retrieval objective.

        Args:
            mu (Tensor): mean vectors, shape ``[B, D]``.
            sigma (Tensor): log-standard-deviation, shape ``[B, D]``.
                Note: the actual std is ``exp(sigma)``, ensuring positivity.
            nums (int): number of Monte-Carlo samples to draw per vector.

        Returns:
            Tensor: sampled features, shape ``[B, nums, D]``.
        """
        # epsilon ~ N(0, I): independent standard normal noise
        eps = torch.randn(mu.size(0), nums, mu.size(1), dtype=mu.dtype, device=mu.device)
        # z = mu + std * eps, with std = exp(sigma)
        # unsqueeze(1) aligns sigma from [B, D] to [B, 1, D] for broadcasting
        samples = eps.mul(torch.exp(sigma.unsqueeze(1))).add_(mu.unsqueeze(1))
        return samples

    def forward(self, feat, mask=None):
        """Compute Gaussian parameters (mu, sigma) for the input token sequence.

        Args:
            feat (Tensor): input token features, shape ``[B, N, D]``.
                B = batch, N = sequence length, D = embed_dim.
            mask (Tensor, optional): validity mask, shape ``[B, N]``.
                Passed through to the attention pooling inside
                ``MultiHeadSelfAttention``.

        Returns:
            tuple of two Tensors:
                - **mu**    (Tensor): L2-normalized mean, shape ``[B, D]``.
                - **sigma** (Tensor): log-standard-deviation, shape ``[B, D]``.
        """
        # ------------------------------------------------------------------
        # Mu branch: mean vector
        # ------------------------------------------------------------------
        # 1) Self-attention + pooling collapses [B, N, D] -> [B, D]
        # 2) MLP projects back to the same dimension
        # 3) LayerNorm stabilizes activations
        # 4) L2 normalization constrains mu to the unit sphere
        out_mu = self.mlp_mu(self.attention_mu(feat, mask))
        out_mu = self.ln_1(out_mu)
        out_mu = self.l2_normalize(out_mu)

        # ------------------------------------------------------------------
        # Sigma branch: log-standard-deviation vector
        # ------------------------------------------------------------------
        # Same pipeline as mu but without L2 normalization. The magnitude is
        # unconstrained (can be positive or negative) because the actual std
        # is exp(sigma), which is always positive.
        out_sigma = self.mlp_sigma(self.attention_sigma(feat, mask))
        out_sigma = self.ln_2(out_sigma)

        return out_mu, out_sigma


# =============================================================================
# Standalone test script (executes when module is run directly)
# =============================================================================
# Verifies shape consistency end-to-end:
#   1) ProbEmbedModule forward : [B, N, D] -> ([B, D], [B, D])
#   2) sample_gaussian          : ([B, D], [B, D]) -> [B, nums, D]
# =============================================================================
# model = ProbEmbedModule()

# batch_size, nums_size, embed_size = 32, 32, 512
# t_feat = torch.randn(batch_size, nums_size, embed_size)
# t_mask = t_feat.new_ones(t_feat.size(0), t_feat.size(1))
# print("1--->>>", t_feat.shape, t_mask.shape)
# mu, sigma = model(t_feat, t_mask)
# print("2--->>>", mu.shape, sigma.shape)
# t_prob = model.sample_gaussian(mu, sigma, nums=1000)
# print("3--->>>", t_prob.shape)

# Historical usage pattern from the original retrieval model:
#
#   out = self.agg_text(t_feat)
#   out = self.prob_text(t_feat, out)
#   mu, sigma = out['mu'], out['sigma']
#   probs = self.sample_gaussian(mu, sigma, self.sample_text_n)
#   output = {'mu': mu, 'sigma': sigma, 'probs': probs}
#   return output
