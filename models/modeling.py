import os
from collections import OrderedDict
from types import SimpleNamespace
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# Probabilistic embedding: maps deterministic vectors to Gaussian distributions (mu, sigma)
from .module_prob import ProbEmbedModule
# Evidential Deep Learning: regularizes similarity matrices with Dirichlet uncertainty
from .module_edl import edl_module
# OpenAI CLIP backbone (vision + text encoder) and weight-conversion helpers
from .module_clip import CLIP, convert_weights, _PT_NAME
# Temporal Transformer for frame-level sequential aggregation
from .module_cross import Transformer as TransformerClip
# Utilities: LayerNorm, distributed all-gather ops, contrastive & KL losses
from .until_module import LayerNorm, AllGather, AllGather2, CrossEn, KL, KLdivergence

# =============================================================================
# Distributed training custom autograd functions
# =============================================================================
# allgather: all-reduce across GPUs with standard gradient behavior
# allgather2: all-reduce across GPUs with SUM-reduced gradient (used for certain
#             contrastive learning setups where gradients need to be averaged
#             across the distributed group)
allgather = AllGather.apply
allgather2 = AllGather2.apply


class ResidualLinear(nn.Module):
    """Residual linear block: x + (Linear -> ReLU -> Linear)(x)."""

    def __init__(self, d_int: int):
        super(ResidualLinear, self).__init__()
        self.fc_relu = nn.Sequential(
            nn.Linear(d_int, d_int),
            nn.ReLU(inplace=True),
            nn.Linear(d_int, d_int),
        )

    def forward(self, x):
        x = x + self.fc_relu(x)
        return x


class Model(nn.Module):
    """Core model for TSI-TVR (Temporal-Spatial Interaction for Text-Video Retrieval).

    This model extracts multi-granularity text and video features via CLIP, then
    computes text-to-video similarity through three parallel spatial interaction
    branches at different granularities, fused by learnable weights.

    Architecture overview:
        1. CLIP Encoder: extracts sentence/word-level text features and frame/patch-level
           visual features;
        2. Temporal Aggregation (optional): seqLSTM or seqTransf across frame sequences;
        3. Multi-granularity Spatial Interaction:
           - qs-vf: query-sentence  vs video-frame  (global vs global);
           - qw-vf: query-word     vs video-frame  (local  vs global);
           - qp-vp: query-phrase   vs video-patch  (probabilistic vs probabilistic);
        4. Learnable Fusion: a 3-D learnable weight vector (softmax-normalized) scales
           the three similarity matrices into a single unified similarity matrix;
        5. Probabilistic Module: ProbEmbedModule maps deterministic features to Gaussian
           latent distributions (mu, sigma), sampled for stochastic matching in qp-vp;
        6. EDL Module: edl_module adds uncertainty-aware regularization to the fused
           similarity matrix;
        7. Total Loss: symmetric contrastive loss + alpha * EDL loss + beta * KL loss.
    """

    def __init__(self, config):
        """Initialize all model modules.

        Key config attributes:
            - interaction: interaction type string.
            - agg_module: video frame aggregation mode, one of 'meanP' (mean pooling),
              'seqLSTM', or 'seqTransf'.
            - base_encoder: CLIP backbone variant, e.g., "ViT-B/32".
            - num_hidden_layers: number of Transformer layers for seqTransf agg_module.
            - max_words: maximum number of words per text sequence.
            - max_frames: maximum number of video frames.
            - alpha: weight for EDL losses.
            - beta: weight for KL divergence between sampled distributions.
            - save_frames: number of top-relevance frames retained after compact
              (default max_frames // 2). Currently unused in this forward implementation.

        Weight loading strategy:
            1. Construct CLIP with dimensions auto-derived from pretrained weights.
            2. Call ``self.apply(self.init_weights)`` to randomly initialize new modules.
            3. Load pretrained CLIP weights into the backbone (strict=False allows new keys).
            4. (Optional) Warm-start: copy CLIP positional embeddings / Transformer layers
               into frame_position_embeddings and transformerClip for training stability.
        """
        super(Model, self).__init__()

        self.config = config
        self.interaction = config.interaction
        self.agg_module = getattr(config, 'agg_module', 'meanP')
        backbone = getattr(config, 'base_encoder', "ViT-B/32")

        assert backbone in _PT_NAME
        # Pretrained CLIP weights are searched in ./models/, then in the project root
        model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), _PT_NAME[backbone])
        if not os.path.exists(model_path):
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _PT_NAME[backbone])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
        try:
            model = torch.jit.load(model_path, map_location="cpu").eval()
            state_dict = model.state_dict()
        except RuntimeError:
            state_dict = torch.load(model_path, map_location="cpu")

        # =================================================================
        # Auto-derive CLIP hyperparameters from pretrained weight shapes
        # =================================================================
        # ``visual.conv1`` output channel = ViT hidden dimension (e.g. 768 for ViT-B/32)
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        # Count attention layers by matching key suffix pattern
        vision_layers = len(
            [k for k in state_dict.keys() if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")])
        # Patch embedding kernel size = patch size (e.g. 32)
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        # [CLS] token at position 0 in positional_embedding, so subtract 1 before sqrt
        grid_size = round((state_dict["visual.positional_embedding"].shape[0] - 1) ** 0.5)
        image_resolution = vision_patch_size * grid_size  # e.g. 224 for ViT-B/32

        # Text encoder dimensions
        embed_dim = state_dict["text_projection"].shape[1]
        context_length = state_dict["positional_embedding"].shape[0]
        vocab_size = state_dict["token_embedding.weight"].shape[0]
        transformer_width = state_dict["ln_final.weight"].shape[0]
        transformer_heads = transformer_width // 64
        transformer_layers = len(set(k.split(".")[2] for k in state_dict if k.startswith(f"transformer.resblocks")))

        # Initialize CLIP backbone (vision encoder + text encoder)
        self.clip = CLIP(embed_dim, image_resolution, vision_layers, vision_width, vision_patch_size,
                         context_length, vocab_size, transformer_width, transformer_heads, transformer_layers)

        if torch.cuda.is_available():
            # Convert CLIP weights to fp16 for mixed-precision training speed-up
            convert_weights(self.clip)

        # Cross-modal Transformer configuration for seqTransf frame aggregation.
        # We use SimpleNamespace so we can pass a lightweight config object
        # without defining a full class for the temporal Transformer.
        cross_config = SimpleNamespace(**{
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 512,
            "initializer_range": 0.02,
            "intermediate_size": 2048,
            "max_position_embeddings": 128,
            "num_attention_heads": 8,
            "num_hidden_layers": 4,
            "vocab_size": 512,
            "soft_t": 0.07,
        })
        # Override with actual CLIP-derived values so temporal embeddings / hidden dims match
        cross_config.max_position_embeddings = context_length
        cross_config.hidden_size = transformer_width
        self.cross_config = cross_config

        # =================================================================
        # Optional temporal aggregation module for video frames
        # =================================================================
        if self.agg_module in ["seqLSTM", "seqTransf"]:
            # Frame-level positional embeddings for temporal ordering
            self.frame_position_embeddings = nn.Embedding(cross_config.max_position_embeddings,
                                                          cross_config.hidden_size)
            if self.agg_module == "seqTransf":
                # Multi-head Transformer for temporal modeling across frame sequences
                self.transformerClip = TransformerClip(width=transformer_width,
                                                       layers=config.num_hidden_layers,
                                                       heads=transformer_heads)
            if self.agg_module == "seqLSTM":
                # Unidirectional LSTM for temporal modeling across frame sequences
                self.lstm_visual = nn.LSTM(input_size=cross_config.hidden_size,
                                           hidden_size=cross_config.hidden_size,
                                           batch_first=True, bidirectional=False, num_layers=1)

        # Loss functions
        self.loss_fct = CrossEn(config)  # Symmetric contrastive loss (cross-entropy)
        # NOTE: self.loss_kl is instantiated but NOT used in forward().
        # The KL term in the training objective (beta * loss_kl) actually comes from
        # self.loss_prob_kl operating on the Gaussian posteriors (see Region M3).
        # This module is kept for backward compatibility with older training loops
        # that may reference it explicitly.
        self.loss_kl = KL(config)        # KL divergence loss (historical, unused in current forward)

        self.apply(self.init_weights)  # Init new modules before loading pretrained weights
        self.clip.load_state_dict(state_dict, strict=False)

        # =================================================================
        # Post-CLIP initialization (probabilistic & learnable components)
        # =================================================================
        # Re-derive embed_dim because the loaded weights supersede the above inference
        embed_dim = state_dict["text_projection"].shape[1]
        self.max_words = config.max_words
        self.max_frames = config.max_frames
        self.alpha = config.alpha
        self.beta = config.beta

        # EDL modules for uncertainty-aware loss (feature-level + prob-level)
        self.sims_edl = edl_module(kl_weight=0.01)
        self.text_prob = ProbEmbedModule()   # text -> Gaussian (mu, sigma)
        self.video_prob = ProbEmbedModule()  # video -> Gaussian (mu, sigma)
        self.prob_edl = edl_module(kl_weight=0.01)

        # KL divergence between Gaussian posteriors (analytical diagonal form).
        # This is the KL module actually exercised in forward() (see Region M3).
        self.loss_prob_kl = KLdivergence()

        # =================================================================
        # Learnable per-granularity attention-aggregation weight networks
        # =================================================================
        # Each network: Linear(embed_dim, embed_dim*2) -> ReLU -> Linear(embed_dim*2, 1)
        # The scalar weights are softmax-normalized over valid tokens and used as
        # coefficients in the symmetric attention similarity functions.
        self.qs_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.qw_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.vf_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.qp_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))
        self.vp_feat_w = nn.Sequential(nn.Linear(embed_dim, embed_dim * 2), nn.ReLU(), nn.Linear(embed_dim * 2, 1))

        # =================================================================
        # Learnable multi-branch similarity fusion weights
        # =================================================================
        # A 3-D learnable parameter that produces a softmax-normalized weight vector
        # over the three similarity branches: [w0, w1, w2] where
        #   w0 scales sims_qs_vf (global vs global),
        #   w1 scales sims_qw_vf (local vs global),
        #   w2 scales sims_qp_vp (probabilistic vs probabilistic).
        # Initialized to ones => initial weights are uniform (1/3 each).
        # This replaces the previous fixed arithmetic average with an adaptive,
        # data-driven fusion strategy.
        self.sims_weights = nn.Parameter(torch.ones(3))

        # =================================================================
        # Warm-start trick: port CLIP pretrained embeddings into agg modules
        # =================================================================
        new_state_dict = OrderedDict()

        if self.agg_module in ["seqLSTM", "seqTransf"]:
            contain_frame_position = False
            for key in state_dict.keys():
                if key.find("frame_position_embeddings") > -1:
                    contain_frame_position = True
                    break
            if contain_frame_position is False:
                for key, val in state_dict.items():
                    if key == "positional_embedding":
                        # Copy text positional embeddings to frame position embeddings
                        # so the temporal module starts with meaningful position encodings.
                        new_state_dict["frame_position_embeddings.weight"] = val.clone()
                        continue
                    # Copy first ``config.num_hidden_layers`` CLIP Transformer layers
                    # into transformerClip to preserve learned attention patterns.
                    if self.agg_module in ["seqTransf"] and key.find("transformer.resblocks") == 0:
                        num_layer = int(key.split(".")[2])
                        if num_layer < config.num_hidden_layers:
                            # Map CLIP key "transformer.resblocks.N.X" to "transformerClip.resblocks.N.X"
                            new_state_dict[key.replace("transformer.", "transformerClip.")] = val.clone()
                            continue
        self.load_state_dict(new_state_dict, strict=False)

    # =====================================================================
    #  Forward pass
    # =====================================================================
    def forward(self, query, query_word_mask, video, video_frame_mask, idx=None, global_step=0):
        """Full training forward pass.

        Pipeline (unified similarity + single loss):
            1. Feature extraction: CLIP encodes query into qs (sentence) and qw (word)
               features; encodes video into vf (frame) and vp (patch) features.
            2. Compute three parallel similarity matrices:
               sims_qs_vf (global vs global), sims_qw_vf (local vs global), sims_qp_vp
               (probabilistic phrase vs probabilistic patch via Gaussian sampling).
            3. Fuse: softmax-normalized learnable weights combine the three matrices
               into a single unified similarity matrix ``sims``.
            4. Loss: symmetric contrastive loss on ``sims``, plus EDL regularization,
               plus KL divergence between the Gaussian posteriors of text and video.
            5. Total: loss_sims + alpha * loss_edls + beta * loss_kl.

        Args:
            query (Tensor): query text token IDs, shape [a, L]. May have outer batch dims.
            query_word_mask (Tensor): query text padding mask, [a, L]. Non-zero = valid.
            video (Tensor): video frames. Two layouts:
                [b, n_v, channel, h, w] (standard clip) or
                [b, pair, bs, ts, channel, h, w] (nested paired batch).
            video_frame_mask (Tensor): frame validity mask, [b, n_v].
            idx (Tensor, optional): sample indices. Reserved, currently unused.
            global_step (int, optional): current training step. Reserved, unused.

        Returns:
            Tensor: scalar total training loss when self.training is True.
            None: in eval mode (use get_similarity_logits for inference).

        Shape conventions in this method:
            - a: number of text queries (after all-gather)
            - b: number of video samples (after all-gather)
            - w: number of query words
            - f: number of frames
            - p: number of patches ( == f * patches_per_frame )
        """
        # -------------------------------------------------------------------
        # Flatten any outer paired-batch dimensions (e.g. pair, bs, ts)
        # -------------------------------------------------------------------
        query = query.reshape(-1, query.shape[-1])
        query_word_mask = query_word_mask.reshape(-1, query_word_mask.shape[-1])

        video = torch.as_tensor(video).float()
        video_frame_mask = video_frame_mask.reshape(-1, video_frame_mask.shape[-1])

        # video layout dispatch: 5-D or 7-D
        if len(video.size()) == 5:
            b, n_v, d, h, w = video.shape
            video = video.reshape(b * n_v, d, h, w)
        else:
            b, pair, bs, ts, channel, h, w = video.shape
            video = video.reshape(b * pair * bs * ts, channel, h, w)

        # ===================================================================
        # Step 1: multi-granularity feature extraction via CLIP
        # ===================================================================
        qs_feat, qw_feat = self.get_text_feat(query, query_word_mask)       # [a, d], [a, w, d]
        vf_feat, vp_feat = self.get_video_feat(video, video_frame_mask)     # [b, f, d], [b, p, d]

        # -------------------------------------------------------------------
        # Build granularity masks for downstream interaction functions
        # -------------------------------------------------------------------
        # qs_mask is always valid: each query sentence has exactly 1 token
        qs_mask = qs_feat.new_ones(qs_feat.size(0), 1)      # [a, 1]
        qw_mask = query_word_mask                           # [a, w]
        vf_mask = video_frame_mask                          # [b, f]

        # Derive patch mask from frame mask by tiling each frame into patches_per_frame
        # and then flattening. For ViT-B/32 224x224, patches_per_frame = 49.
        # Guard against division by zero (vf_feat.size(1)==0) in degenerate empty batches.
        patches_per_frame = vp_feat.size(1) // vf_feat.size(1) if vf_feat.size(1) > 0 else 0
        vp_mask = (video_frame_mask
                   .unsqueeze(-1)
                   .expand(-1, -1, patches_per_frame)
                   .reshape(video_frame_mask.size(0), -1))

        # Memory layout optimization: contiguous before the expensive gather ops
        qs_feat, qw_feat = qs_feat.contiguous(), qw_feat.contiguous()
        qs_mask, qw_mask = qs_mask.contiguous(), qw_mask.contiguous()
        vf_feat, vp_feat = vf_feat.contiguous(), vp_feat.contiguous()
        vf_mask, vp_mask = vf_mask.contiguous(), vp_mask.contiguous()

        # ===================================================================
        # Step 2: Distributed all-gather for cross-GPU contrastive training
        # ===================================================================
        # Positive pairs may live on different GPUs. All-gather both text and video
        # features so every GPU sees the full mini-batch for InfoNCE computation.
        qs_feat, qw_feat, qs_mask, qw_mask = [allgather(x, self.config) for x in [qs_feat, qw_feat, qs_mask, qw_mask]]
        vf_feat, vp_feat, vf_mask, vp_mask = [allgather(x, self.config) for x in [vf_feat, vp_feat, vf_mask, vp_mask]]
        # Barrier is required because allgather returns async all-reduce handles;
        # we must wait until all workers have completed communication before any
        # downstream computation that consumes the gathered tensors.
        torch.distributed.barrier()

        # -------------------------------------------------------------------
        # Dimension aliases (for readability in similarity functions)
        # -------------------------------------------------------------------
        a, s, w = qs_feat.size(0), 1, qw_feat.size(1)
        b, f, p = vf_feat.size(0), vf_feat.size(1), vp_feat.size(1)

        # CLIP temperature logit_scale is stored as log(1 / temperature); exponentiate
        # here to get the scaling factor applied to similarity logits before softmax.
        logit_scale = self.clip.logit_scale.exp()

        # ===================================================================
        # Region M1: Deterministic Feature-Level Matching
        # ===================================================================
        # qs-vf: global sentence vs global frame similarity
        sims_qs_vf = self.qs_and_vf(qs_feat, qs_mask, vf_feat, vf_mask)   # [a, b]
        # qw-vf: local word vs global frame similarity
        sims_qw_vf = self.qw_and_vf(qw_feat, qw_mask, vf_feat, vf_mask)   # [a, b]

        # ===================================================================
        # Region M2: Probabilistic-Level Matching
        # ===================================================================
        # ProbEmbedModule maps deterministic features into Gaussian latent distributions.
        # We sample 50 times to obtain stochastic phrase/patch representations.
        # The choice of 50 is a Monte Carlo estimate: empirically sufficient to
        # approximate the expected similarity under the Gaussian posteriors without
        # excessive memory overhead (50 x batch_size x num_tokens).

        # Text probabilistic branch
        qw_prob_mu, qw_prob_sigma = self.text_prob(qw_feat, qw_mask)
        qw_prob = self.text_prob.sample_gaussian(qw_prob_mu, qw_prob_sigma, nums=10)
        # After sampling, all 50 draws are equally valid; replace mask with all-ones
        qw_mask = qw_prob.new_ones(qw_prob.size(0), qw_prob.size(1))   # [a, nums]

        # Video probabilistic branch
        vf_prob_mu, vf_prob_sigma = self.video_prob(vf_feat, vf_mask)
        vf_prob = self.video_prob.sample_gaussian(vf_prob_mu, vf_prob_sigma, nums=10)
        vf_mask = vf_prob.new_ones(vf_prob.size(0), vf_prob.size(1))   # [b, nums]

        # qp-vp: probabilistic phrase vs probabilistic patch similarity
        sims_qp_vp = self.qp_and_vp(qw_prob, qw_mask, vf_prob, vf_mask)  # [a, b]

        # ===================================================================
        # Region M3: Learnable multi-branch fusion & unified loss
        # ===================================================================
        # Softmax-normalize the 3-D learnable weight vector so weights sum to 1.
        # Each weight controls the contribution of one similarity branch to the
        # fused similarity matrix. Initialized uniform => starts as arithmetic mean.
        sims_weights = torch.softmax(self.sims_weights, dim=0)

        # Weighted combination of the three similarity branches into one matrix.
        # This unified matrix carries information from all granularities and is
        # the single source for contrastive and EDL losses.
        sims = (sims_weights[0] * sims_qs_vf +
                sims_weights[1] * sims_qw_vf +
                sims_weights[2] * sims_qp_vp)

        # Symmetric contrastive loss (InfoNCE) on the fused similarity.
        # Bidirectional: text->video + video->text, averaged.
        loss_sims = (self.loss_fct(sims * logit_scale) + self.loss_fct(sims.T * logit_scale)) / 2.0

        # EDL uncertainty regularization on the fused similarity matrix.
        # Encourages diagonal (positive-pair) evidence to be high and off-diagonal
        # evidence to be low via Dirichlet distribution modeling.
        # The second return value is per-pair uncertainty/evidence metrics; it is
        # discarded here because the overall loss only needs the aggregated scalar.
        loss_edls, _ = self.sims_edl(sims)

        # KL divergence between text and video posterior distributions.
        # KLdivergence uses the analytical tractable form for diagonal Gaussians.
        loss_kl = self.loss_prob_kl(qw_prob, qw_prob_sigma, vf_prob, vf_prob_sigma)

        # ===================================================================
        # Total loss: contrastive + uncertainty reg + posterior alignment
        # ===================================================================
        # total_loss = loss_sims + self.alpha * loss_edls + self.beta * loss_kl
        total_loss = loss_sims + self.alpha * loss_edls + self.beta * loss_kl

        if self.training:
            return total_loss
        else:
            return None

    # =====================================================================
    # Feature extraction helpers
    # =====================================================================
    def get_text_feat(self, text_ids, text_mask):
        """Extract multi-granularity text features through CLIP.

        CLIP text encoder returns two representations when return_hidden=True:
          - the final layer's pooled representation (sentence level);
          - all hidden states (word level).

        Args:
            text_ids (Tensor): text token IDs, shape [bs_pair, L].
                bs_pair may be multiplied by paired-sampling factors.
            text_mask (Tensor): text attention mask, shape [bs_pair, L].
                Non-zero positions are valid tokens.

        Returns:
            tuple of two Tensors:
                - s_feat: sentence-level features, [bs_pair, embed_dim].
                - w_feat: word-level per-token features, [bs_pair, num_words, embed_dim].
        """
        # Flatten any (pair, bs, ts, ...) outer dimensions
        text_ids = text_ids.reshape(-1, text_ids.shape[-1])
        text_mask = text_mask.reshape(-1, text_mask.shape[-1])

        bs_pair = text_ids.size(0)
        # return_hidden=True yields both the pooled sentence vector and the full
        # token-level hidden states (needed for the qw-vf and qp-vp branches).
        s_feat, w_feat = self.clip.encode_text(text_ids, return_hidden=True, mask=text_mask)
        # .float() forces fp32 in case CLIP was converted to fp16
        s_feat = s_feat.float().reshape(bs_pair, s_feat.size(-1))
        w_feat = w_feat.float().reshape(bs_pair, -1, w_feat.size(-1))
        return s_feat, w_feat

    def get_video_feat(self, video, video_mask):
        """Extract multi-granularity video features through CLIP.

        CLIP vision encoder returns two representations when return_hidden=True:
          - the [CLS] token at position 0 (frame level);
          - all patch tokens at positions 1..N (patch level).

        Args:
            video (Tensor): video frame pixels, [bs_pair * n_v, C, H, W]
                after optional reshaping in inference mode.
            video_mask (Tensor): frame validity mask, [bs_pair, n_v].

        Returns:
            tuple of two Tensors:
                - f_feat: frame-level [CLS] features, [bs_pair, num_frames, embed_dim].
                - p_feat: patch-level features, [bs_pair, num_patches, embed_dim].
                  For ViT-B/32 224x224, num_patches = num_frames * 49.
        """
        if not self.training:
            # In eval mode downstream code may pass the raw nested-batch layout.
            # We mirror the same flattening logic that forward() performs so that
            # both training and inference see identical feature shapes.
            video_mask = video_mask.reshape(-1, video_mask.shape[-1])
            video = torch.as_tensor(video).float()
            if len(video.size()) == 5:
                b, n_v, d, h, w = video.shape
                video = video.reshape(b * n_v, d, h, w)
            else:
                b, pair, bs, ts, channel, h, w = video.shape
                video = video.reshape(b * pair * bs * ts, channel, h, w)

        bs_pair, n_v = video_mask.size()
        f_feat, p_feat = self.clip.encode_image(video, return_hidden=True, mask=video_mask)
        f_feat = f_feat.float().reshape(bs_pair, -1, f_feat.size(-1))
        f_feat = self.agg_video_feat(f_feat, video_mask, self.agg_module)
        p_feat = p_feat.float().reshape(bs_pair, -1, p_feat.size(-1))
        return f_feat, p_feat

    def agg_video_feat(self, video_feat, video_mask, agg_module):
        """Aggregate frame-level features into a unified temporal representation.

        Supports three aggregation strategies:
            - "None"   : identity (no aggregation).
            - "seqLSTM": unidirectional LSTM with residual connection.
                         Uses pack_padded_sequence to handle variable-length inputs.
            - "seqTransf": multi-head Transformer encoder with positional
                         embeddings and residual connection.

        Args:
            video_feat (Tensor): frame features, [bs, num_frames, d].
            video_mask (Tensor): frame validity mask, [bs, num_frames].
                Vectors of 1 (valid) / 0 (pad).
            agg_module (str): one of "None", "seqLSTM", "seqTransf".

        Returns:
            Tensor: aggregated frame features, [bs, num_frames, d].
        """
        video_feat = video_feat.contiguous()
        if agg_module == "None":
            pass
        elif agg_module == "seqLSTM":
            # Sequential type: LSTM
            video_feat_original = video_feat
            # Pack variable-length sequences; skip pad positions during LSTM pass.
            # enforce_sorted=False: PyTorch internally sorts by length.
            video_feat = pack_padded_sequence(video_feat, torch.sum(video_mask, dim=-1).cpu(),
                                              batch_first=True, enforce_sorted=False)
            video_feat, _ = self.lstm_visual(video_feat)
            # flatten_parameters() consolidates weight data to avoid CUDNN warnings.
            # Call only in training to avoid data-dependent control flow in eval.
            if self.training:
                self.lstm_visual.flatten_parameters()
            video_feat, _ = pad_packed_sequence(video_feat, batch_first=True)
            # Pad the tail if output is shorter than input (some batches have more frames)
            video_feat = torch.cat(
                (video_feat, video_feat_original[:, video_feat.size(1):, ...].contiguous()), dim=1)
            # Residual adds stability; unidirectional LSTM otherwise discards raw CLIP reps
            video_feat = video_feat + video_feat_original
        elif agg_module == "seqTransf":
            # Sequential type: Transformer Encoder
            video_feat_original = video_feat
            seq_length = video_feat.size(1)
            # Build absolutely-positioned embeddings for each frame index
            position_ids = torch.arange(seq_length, dtype=torch.long, device=video_feat.device)
            position_ids = position_ids.unsqueeze(0).expand(video_feat.size(0), -1)
            frame_position_embeddings = self.frame_position_embeddings(position_ids)
            video_feat = video_feat + frame_position_embeddings

            # Convert video_mask (1=valid, 0=pad) to large negative attention mask.
            # -1000000.0 approximates -inf while staying fp32-safe.
            extended_video_mask = (1.0 - video_mask.unsqueeze(1)) * -1000000.0
            # Expand to (bs, num_frames, num_frames) so each query position attends
            # to the same set of key positions (standard encoder self-attention mask).
            extended_video_mask = extended_video_mask.expand(-1, video_mask.size(1), -1)

            # TransformerClip expects LND format: (seq_len, batch, dim)
            video_feat = video_feat.permute(1, 0, 2)  # NLD -> LND
            video_feat = self.transformerClip(video_feat, extended_video_mask)
            video_feat = video_feat.permute(1, 0, 2)  # LND -> NLD
            # Residual preserves the original spatial-patch signal
            video_feat = video_feat + video_feat_original
        return video_feat

    # =====================================================================
    # Normalization & similarity helpers
    # =====================================================================
    def norm(self, feat):
        """Apply L2 normalization along the last (feature) dimension.

        The epsilon 1e-8 prevents division by zero for all-zero vectors.

        Args:
            feat (Tensor): input features, shape [..., d].

        Returns:
            Tensor: L2-normalized features, same shape as input.
        """
        return feat / (feat.norm(dim=-1, keepdim=True) + 1e-8)

    def qs_and_vf(self, qs_feat, qs_mask, vf_feat, vf_mask):
        """Global-level spatial interaction: query-sentence vs video-frame.

        Symmetric attention mechanism:
            1) qs -> vf: best-matching frame per query via max aggregation;
            2) vf -> qs: weighted frame aggregation.
        Final similarity is the average of both directions.

        Args:
            qs_feat (Tensor): query sentence features, [a, d].
            qs_mask (Tensor): query sentence mask, [a, 1]. Always 1 for non-empty queries.
            vf_feat (Tensor): video frame features, [b, f, d].
            vf_mask (Tensor): video frame mask, [b, f]. 1=valid, 0=pad.

        Returns:
            Tensor: similarity matrix, [a, b].
        """
        # Step 1: compute a learnable scalar weight for each frame
        # vf_feat_w in (b, f); masked entries become near -inf so softmax assigns zero prob
        vf_feat_w = self.vf_feat_w(vf_feat).squeeze(-1)  # [b, f]
        vf_feat_w = vf_feat_w.masked_fill((1 - vf_mask).to(torch.bool), float(-9e15))
        vf_feat_w = torch.softmax(vf_feat_w, dim=-1)

        # Step 2: pairwise cosine-like similarity [a, b, f]
        # einsum("ad,bfd->abf") broadcasts dot products across the batch
        sims_qs_vf = torch.einsum("ad,bfd->abf", [self.norm(qs_feat), self.norm(vf_feat)])
        # Zero-out padded frame contributions so .max() only considers real frames
        sims_qs_vf = torch.einsum('abf,bf->abf', [sims_qs_vf, vf_mask])

        # Direction: query -> video. For each (query, clip) take the top frame.
        sims_qs2vf, _ = sims_qs_vf.max(dim=-1)  # [a, b]

        # Direction: video -> query. Weighted-sum of all frame contributions.
        # Aggregation weights vf_feat_w are learnable and normalized.
        sims_vf2qs = torch.einsum('abf,bf->ab', [sims_qs_vf, vf_feat_w])  # [a, b]

        # Symmetric average
        sims_qs_vf = (sims_qs2vf + sims_vf2qs) / 2.0
        return sims_qs_vf

    def qw_and_vf(self, qw_feat, qw_mask, vf_feat, vf_mask):
        """Local-to-global spatial interaction: query-word vs video-frame.

        Symmetric attention mechanism:
            1) qw -> vf: best-matching frame per word, then word-weighted aggregate;
            2) vf -> qw: best-matching word per frame, then frame-weighted aggregate.
        Final similarity is the average of both directions.

        Args:
            qw_feat (Tensor): query word features, [a, w, d].
            qw_mask (Tensor): query word padding mask, [a, w].
            vf_feat (Tensor): video frame features, [b, f, d].
            vf_mask (Tensor): video frame mask, [b, f].

        Returns:
            Tensor: similarity matrix, [a, b].
        """
        # Learnable word aggregation weights (masked softmax)
        qw_feat_w = self.qw_feat_w(qw_feat).squeeze(-1)  # [a, w]
        qw_feat_w = qw_feat_w.masked_fill((1 - qw_mask).to(torch.bool), float(-9e15))
        qw_feat_w = torch.softmax(qw_feat_w, dim=-1)

        # Learnable frame aggregation weights (masked softmax)
        vf_feat_w = self.vf_feat_w(vf_feat).squeeze(-1)  # [b, f]
        vf_feat_w = vf_feat_w.masked_fill((1 - vf_mask).to(torch.bool), float(-9e15))
        vf_feat_w = torch.softmax(vf_feat_w, dim=-1)

        # Pairwise similarity tensor [a, b, w, f]
        # Use normalized features to strip magnitude bias
        sims_qw_vf = torch.einsum("awd,bfd->abwf", [self.norm(qw_feat), self.norm(vf_feat)])
        sims_qw_vf = torch.einsum('abwf,aw->abwf', [sims_qw_vf, qw_mask])
        sims_qw_vf = torch.einsum('abwf,bf->abwf', [sims_qw_vf, vf_mask])

        # qw -> vf: max over f per word, then weighted-sum over words
        sims_qw2vf, _ = sims_qw_vf.max(dim=-1)  # [a, b, w]
        sims_qw2vf = torch.einsum('abw,aw->ab', [sims_qw2vf, qw_feat_w])

        # vf -> qw: max over w per frame, then weighted-sum over frames
        sims_vf2qw, _ = sims_qw_vf.max(dim=-2)  # [a, b, f]
        sims_vf2qw = torch.einsum('abf,bf->ab', [sims_vf2qw, vf_feat_w])

        # Symmetric average
        sims_qw_vf = (sims_qw2vf + sims_vf2qw) / 2.0
        return sims_qw_vf

    def qp_and_vp(self, qp_feat, qp_mask, vp_feat, vp_mask):
        """Probabilistic spatial interaction: query-phrase vs video-patch.

        This method is structurally identical to qw_and_vf, but operates on the
        probabilistically-sampled phrase and patch tokens produced by ProbEmbedModule.

        Symmetric attention mechanism:
            1) qp -> vp: each phrase token picks the best-matching patch,
               then tokens are aggregated by learnable phrase weights.
            2) vp -> qp: each patch picks the best-matching phrase,
               then aggregated by learnable patch weights.
        Final similarity is the arithmetic mean.

        Parameters naming convention:
            - qp_* = query-phrase probabilistic features
            - vp_* = video-patch probabilistic features
            - suffix _w = learnable aggregation weight

        Args:
            qp_feat (Tensor): query phrase (sampled) features, [a, p, d].
                Here p is the number of probabilistic samples.
            qp_mask (Tensor): query phrase mask, [a, p]. Always all-ones after sampling.
            vp_feat (Tensor): video patch (sampled) features, [b, v, d].
            vp_mask (Tensor): video patch mask, [b, v].

        Returns:
            Tensor: similarity matrix, [a, b].

        Note:
            The einsum code below uses awd / bpd notation for consistency with
            qw_and_vf, but here w and p refer to the probabilistic sample dimension.
        """
        # Step A: learnable aggregation weights for query-phrase tokens
        qp_feat_w = self.qp_feat_w(qp_feat).squeeze(-1)  # [a, p]
        # Masked fill ensures pad positions get near -inf, so softmax assigns zero prob
        qp_feat_w = qp_feat_w.masked_fill((1 - qp_mask).to(torch.bool), float(-9e15))
        qp_feat_w = torch.softmax(qp_feat_w, dim=-1)

        # Step B: learnable aggregation weights for video-patch tokens
        vp_feat_w = self.vp_feat_w(vp_feat).squeeze(-1)  # [b, v]
        vp_feat_w = vp_feat_w.masked_fill((1 - vp_mask).to(torch.bool), float(-9e15))
        vp_feat_w = torch.softmax(vp_feat_w, dim=-1)

        # Step C: pairwise 4-D similarity tensor [a, b, p, v]
        # Holds every (phrase_sample, patch_sample) cosine score
        sims_qp_vp = torch.einsum("awd,bpd->abwp", [self.norm(qp_feat), self.norm(vp_feat)])
        # Apply masks to zero-out invalid entries (should be all-active here)
        sims_qp_vp = torch.einsum('abwp,aw->abwp', [sims_qp_vp, qp_mask])
        sims_qp_vp = torch.einsum('abwp,bp->abwp', [sims_qp_vp, vp_mask])

        # Step D: qp -> vp direction. For each (query, clip, phrase_sample), find best patch.
        sims_qp2vp, _ = sims_qp_vp.max(dim=-1)  # [a, b, p]
        # Then pool phrase samples via learned aggregation weights
        sims_qp2vp = torch.einsum('abw,aw->ab', [sims_qp2vp, qp_feat_w])

        # Step E: vp -> qp direction. For each (query, clip, patch_sample), find best phrase.
        sims_vp2qp, _ = sims_qp_vp.max(dim=-2)  # [a, b, v]
        # Then pool patch samples via learned aggregation weights
        sims_vp2qp = torch.einsum('abp,bp->ab', [sims_vp2qp, vp_feat_w])

        # Symmetric average
        sims_qp_vp = (sims_qp2vp + sims_vp2qp) / 2.0
        return sims_qp_vp

    def get_similarity_logits(self, qs_feat, qs_mask, qw_feat, qw_mask, vf_feat, vf_mask, vp_feat, vp_mask):
        """Compute similarity logits for inference.

        Unlike the training forward, this method skips loss computation and returns
        a single fused similarity matrix for ranking. It computes the three similarity
        branches (qs-vf, qw-vf, qp-vp), fuses them using the same learnable softmax
        weights as training, and returns the unified similarity matrix.

        Args:
            qs_feat (Tensor): query sentence features, [a, d].
            qs_mask (Tensor): query sentence mask, [a, 1].
            qw_feat (Tensor): query word features, [a, w, d].
            qw_mask (Tensor): query word mask, [a, w].
            vf_feat (Tensor): video frame features, [b, f, d].
            vf_mask (Tensor): video frame mask, [b, f].
            vp_feat (Tensor): video patch features, [b, p, d].
            vp_mask (Tensor): video patch mask, [b, p].

        Returns:
            Tensor: fused similarity matrix, [a, b].
        """
        # Deterministic feature-level similarities
        sims_qs_vf = self.qs_and_vf(qs_feat, qs_mask, vf_feat, vf_mask)   # [a, b]
        sims_qw_vf = self.qw_and_vf(qw_feat, qw_mask, vf_feat, vf_mask)   # [a, b]

        # ===================================================================
        # Probabilistic-level similarities
        # ===================================================================
        # Text probabilistic branch: 50 Monte-Carlo samples from Gaussian posterior
        qw_prob_mu, qw_prob_sigma = self.text_prob(qw_feat, qw_mask)
        qw_prob = self.text_prob.sample_gaussian(qw_prob_mu, qw_prob_sigma, nums=50)
        # All 50 samples are valid; replace mask with all-ones
        qw_mask = qw_prob.new_ones(qw_prob.size(0), qw_prob.size(1))     # [a, nums]

        # Video probabilistic branch: 50 Monte-Carlo samples
        vf_prob_mu, vf_prob_sigma = self.video_prob(vf_feat, vf_mask)
        vf_prob = self.video_prob.sample_gaussian(vf_prob_mu, vf_prob_sigma, nums=50)
        vf_mask = vf_prob.new_ones(vf_prob.size(0), vf_prob.size(1))     # [b, nums]

        # qp-vp: probabilistic phrase vs patch similarity
        sims_qp_vp = self.qp_and_vp(qw_prob, qw_mask, vf_prob, vf_mask)  # [a, b]

        # ===================================================================
        # Learnable fusion: same softmax weights as training forward
        # ===================================================================
        # Softmax-normalize the 3-D parameter; weights adaptively balance the three
        # granularity branches based on what the model learns is informative.
        sims_weights = torch.softmax(self.sims_weights, dim=0)
        sims = (sims_weights[0] * sims_qs_vf +
                sims_weights[1] * sims_qw_vf +
                sims_weights[2] * sims_qp_vp)

        return sims

    @property
    def dtype(self):
        """Return the dtype of the first model parameter.

        Falls back to scanning all tensor attributes via _named_members if no
        nn.Parameter objects exist (e.g. for an empty module or a container only
        holding buffers).

        Returns:
            torch.dtype: inferred dtype of the module.
        """
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            def find_tensor_attributes(module: nn.Module):
                tuples = [(k, v) for k, v in module.__dict__.items() if torch.is_tensor(v)]
                return tuples

            gen = self._named_members(get_members_fn=find_tensor_attributes)
            first_tuple = next(gen)
            return first_tuple[1].dtype

    def init_weights(self, module):
        """Initialize weights for newly added modules.

        Applied via ``self.apply(self.init_weights)`` before loading pretrained
        CLIP weights. Only the extensions added by this model file receive random
        initialization; pretrained parameters are overwritten by the loaded state dict.

        Initialization rules:
            - nn.Linear / nn.Embedding: Normal(mean=0.0, std=0.02)
            - nn.Linear bias (if present): zeros
            - LayerNorm weight (gamma): ones
            - LayerNorm bias (beta): zeros

        Args:
            module (nn.Module): submodule to initialize in place.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, LayerNorm):
            # Handle both legacy gamma/beta naming and standard weight/bias
            if 'beta' in dir(module) and 'gamma' in dir(module):
                module.beta.data.zero_()
                module.gamma.data.fill_(1.0)
            else:
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
